import pickle
import pandas as pd
import numpy as np
from random import choice
from tqdm import tqdm, trange
from livelossplot import PlotLosses


def adv_index(list_to_index, list_to_match):
    return [ind for ind, match in enumerate(list_to_index) if match in list_to_match]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_model(load_path):
    with open(load_path, 'rb') as model_input:
        model_instance = pickle.load(model_input)
    return model_instance


def save_model(model_instance, saved_path):
    with open(saved_path + '/model.pkl', 'wb') as model_output:
        pickle.dump(model_instance, model_output, pickle.HIGHEST_PROTOCOL)
    return None


class bprH(object):

    def __init__(self, dim, omega, rho, lambda_u, lambda_v, lambda_b, gamma, num_iter=200, random_state=None):
        """

        :param dim: the dimension of latent vector
        :param omega: parameter controlling alpha_u
        :param rho: parameter controlling alpha_u
        :param lambda_u: hyper-parameter in user matrix's regularization
        :param lambda_v: hyper-parameter in item matrix's regularization
        :param lambda_b: hyper-parameter in item bias's regularization
        :param gamma: SGD step size
        :param random_state:
        """
        self.dim = dim
        self.omega = omega
        self.rho = rho
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_b = lambda_b
        self.gamma = gamma
        self.random_state = random_state
        self.num_iter = num_iter

        self.user_original_id_list = None
        self.item_original_id_list = None
        self.item_list = None
        self.user_list = None
        self.user_purchased_item = None

        self.num_u = None
        self.num_i = None

        self.U = None
        self.V = None
        self.estimation = None

        self.alpha_u = None
        self.S = None

        self.train_data = None
        self.test_data = None

    def auxiliary_target_correlation(self, X, y=None):
        """
        Calculate auxiliary-target correlation C for every user and each types of auxiliary action
        Here we only have one auxiliary action 'V' for 'View'
        :param X: input data
        :param y:
        :return: return alpha_u
        """
        print("Calculate auxiliary-target correlation")
        target_action = 'P'
        auxiliary_action = ['V']
        C_u = dict()
        user_set_bar = tqdm(self.user_list)
        for u in user_set_bar:
            C_u[u] = dict()
            I_t_u = set(X[(X.UserID == u) & (X.Action == target_action)].ItemID)
            # TODO: filtered item set
            for x in auxiliary_action:
                I_a_u = set(X[(X.UserID == u) & (X.Action == x)].ItemID)

                C_u_at = len(I_t_u.intersection(I_a_u)) / len(I_t_u) if len(I_t_u) != 0 else 0
                C_u_ta = len(I_t_u.intersection(I_a_u)) / len(I_a_u) if len(I_a_u) != 0 else 0

                C_u_X = 2 * C_u_at * C_u_ta / (C_u_ta + C_u_at) if C_u_ta + C_u_at != 0 else (1 / self.omega)
                # set final alpha_u to 1 if C_u_ta + C_u_at == 0
                C_u[u][x] = C_u_X

        temp = pd.DataFrame.from_dict(C_u, orient='index')
        # We have only one auxiliary action 'V'
        temp['alpha'] = self.omega * self.rho * temp.V
        alpha_u = temp
        del temp
        alpha_u.reset_index(inplace=True)
        alpha_u.columns = ['UserID', 'V', 'alpha']
        return alpha_u

    def itemset_coselection(self, saved_path, X, y=None):
        """

        :param X:
        :param y:
        :param saved_path:
        :return:
        """
        print("Generate Itemset Coselection")
        S = dict()
        item_set_bar = tqdm(self.item_list)
        for i in item_set_bar:
            S[i] = set()
            U_i = set(X[X.ItemID == i].UserID)
            for j in self.item_list:
                U_j = set(X[X.ItemID == j].UserID)
                if len(U_i.intersection(U_j)) >= 2: S[i].add(j)
        # save coselection list
        with open(saved_path, 'wb') as f:
            pickle.dump(S, f, pickle.HIGHEST_PROTOCOL)

        return S

    def fit(self, X, eval_X, original_item_list, original_user_list, y=None,
            saved_path='data/item-set-coselection.pkl', coselection=False, plot_metric=False):
        # TODO: make sure train and test works with inconsistent user and item list

        # rename user and item
        self.user_original_id_list = sorted(set(original_user_list))
        self.item_original_id_list = sorted(set(original_item_list))

        self.train_data = X.copy()
        self.test_data = eval_X.copy()

        self.train_data.UserID = self.train_data.UserID.apply(lambda x: self.user_original_id_list.index(x))
        self.train_data.ItemID = self.train_data.ItemID.apply(lambda x: self.item_original_id_list.index(x))

        self.test_data.UserID = self.test_data.UserID.apply(lambda x: self.user_original_id_list.index(x))
        self.test_data.ItemID = self.test_data.ItemID.apply(lambda x: self.item_original_id_list.index(x))

        self.item_list = sorted(set([idx[0] for idx in enumerate(self.item_original_id_list)]))
        self.user_list = sorted(set([idx[0] for idx in enumerate(self.user_original_id_list)]))

        self.num_u = len(self.user_list)
        self.num_i = len(self.item_list)

        # Calculate auxiliary-target correlation C for every user and each types of auxiliary action
        self.alpha_u = self.auxiliary_target_correlation(X=self.train_data)

        # Generate item-set based on co-selection
        if coselection:
            self.S = self.itemset_coselection(X=self.train_data, saved_path=saved_path)

        # Initialization of User and Item Matrices
        if self.random_state is not None:
            np.random.seed(self.random_state)
        else:
            np.random.seed(0)

        self.U = np.random.normal(size=(self.num_u, self.dim + 1))
        self.V = np.random.normal(size=(self.dim + 1, self.num_i))
        self.U[:, -1] = 1
        # estimation is U dot V
        self.estimation = np.dot(self.U, self.V)

        # plot loss
        if plot_metric:
            groups = {'Precision@K': ['Precision@5', 'Precision@10'], 'Recall@K': ['Recall@5', 'Recall@10']}
            plot_losses = PlotLosses(groups=groups)

        # Start Iteration
        with trange(self.num_iter) as t:
            for index in t:
                # Description will be displayed on the left
                # t.set_description('ITER %i' % index)

                # Build u, I, J, K
                # uniformly sample a user from U
                u = choice(sorted(set(self.train_data.UserID)))

                # build I
                # uniformly sample a item i from I_u_t
                I_u_t = set(self.train_data[(self.train_data.UserID == u) & (self.train_data.Action == 'P')].ItemID)
                if len(I_u_t) != 0:
                    i = choice(sorted(I_u_t))
                    # build I = I_u_t cap S_i
                    if coselection:
                        I = I_u_t.intersection(self.S[i])
                    else:
                        I = I_u_t
                else:  # if no item in I_u_t, then set I to empty set
                    i = None
                    I = set()

                # build J, since we only have one auxiliary action, we follow the uniform sampling
                I_u_oa = set(self.train_data[(self.train_data.UserID == u) & (
                        self.train_data.Action == 'V')].ItemID) - I_u_t  # TODO: optimize this
                if len(I_u_oa) != 0:
                    j = choice(sorted(I_u_oa))
                    if coselection:
                        J = I_u_oa.intersection(self.S[j])
                    else:
                        J = I_u_oa
                else:  # if no item in I_u_oa, then set J to empty set
                    j = None
                    J = set()

                # build K
                I_u_n = set(self.item_list) - I_u_t - I_u_oa
                if len(I_u_n) != 0:
                    k = choice(sorted(I_u_n))
                    # build K
                    if coselection:
                        K = I_u_n.intersection(self.S[k])
                    else:
                        K = I_u_n
                else:  # if no item in I_u_n, then set K to empty set
                    k = None
                    K = set()

                # calculate intermediate variables
                # get specific alpha_u
                spec_alpha_u = self.alpha_u[self.alpha_u.UserID == u].alpha.values[0]

                U_u = self.U[u, :-1]
                # get r_hat_uIJ and r_hat_uJK
                r_hat_uI = np.average(self.estimation[u, sorted(I)]) if len(
                    I) != 0 else 0
                r_hat_uJ = np.average(self.estimation[u, sorted(J)]) if len(
                    J) != 0 else 0
                r_hat_uK = np.average(self.estimation[u, sorted(K)]) if len(
                    K) != 0 else 0

                r_hat_uIJ = r_hat_uI - r_hat_uJ
                r_hat_uJK = r_hat_uJ - r_hat_uK
                # get V_bar_I, V_bar_J, V_bar_K
                V_bar_I = np.average(self.V[:-1, sorted(I)], axis=1) if len(
                    I) != 0 else np.zeros(
                    shape=self.V[:-1, 0].shape)
                V_bar_J = np.average(self.V[:-1, sorted(J)], axis=1) if len(
                    J) != 0 else np.zeros(
                    shape=self.V[:-1, 0].shape)
                V_bar_K = np.average(self.V[:-1, sorted(K)], axis=1) if len(
                    K) != 0 else np.zeros(
                    shape=self.V[:-1, 0].shape)
                # get b_I, b_J, b_K
                b_I = np.average(self.V[-1, sorted(I)]) if len(I) != 0 else 0
                b_J = np.average(self.V[-1, sorted(J)]) if len(J) != 0 else 0
                b_K = np.average(self.V[-1, sorted(K)]) if len(K) != 0 else 0


                # get derivatives and update

                # NABULA U_u
                df_dUu = sigmoid(- r_hat_uIJ / spec_alpha_u) / spec_alpha_u * (V_bar_I - V_bar_J) + \
                         sigmoid(- r_hat_uJK) * (V_bar_J - V_bar_K)
                dR_dUu = 2 * self.lambda_u * U_u
                # update U_u = U_u + gamma * (df_dUu - dR_dUu)
                norm_nabula_U_u = np.linalg.norm((df_dUu - dR_dUu), ord=2)
                self.U[u, :-1] += self.gamma * (df_dUu - dR_dUu)

                if len(I) != 0:
                    # NABULA V_i
                    df_dbi = sigmoid(- r_hat_uIJ / spec_alpha_u) / (len(I) * spec_alpha_u)
                    dR_dbi = 2 * self.lambda_b * b_I / len(I)
                    df_dVi = df_dbi * U_u
                    dR_dVi = 2 * self.lambda_v * V_bar_I / len(I)

                    norm_nabula_Vi = np.linalg.norm((df_dVi - dR_dVi), ord=2)

                    # update V_i = V_i + gamma * (df_dVi - dR_dVi)
                    self.V[:-1, sorted(I)] += self.gamma * (df_dVi - dR_dVi)[:, None]  # trick: transpose here
                    # update b_i = b_i + gamma * (df_dbi - dR_dbi)
                    self.V[-1, sorted(I)] += self.gamma * (df_dbi - dR_dbi)
                else:
                    norm_nabula_Vi = 0

                if len(J) != 0:
                    # NABULA V_j
                    df_dbj = (- sigmoid(- spec_alpha_u * r_hat_uIJ) / spec_alpha_u + sigmoid(- r_hat_uJK)) / len(J)
                    dR_dbj = 2 * self.lambda_b * b_J / len(J)
                    df_dVj = df_dbj * U_u
                    dR_dVj = 2 * self.lambda_v * V_bar_J / len(J)

                    norm_nabula_Vj = np.linalg.norm((df_dVj - dR_dVj), ord=2)

                    # update V_j = V_j + gamma * (df_dVj - dR_dVj)
                    self.V[:-1, sorted(J)] += self.gamma * (df_dVj - dR_dVj)[:, None]  # trick: transpose here
                    # update b_j = b_j + gamma * (df_dbj - dR_dbj)
                    self.V[-1, sorted(J)] += self.gamma * (df_dbj - dR_dbj)
                else:
                    norm_nabula_Vj = 0

                if len(K) != 0:
                    # NABULA V_k
                    df_dbk = - sigmoid(- r_hat_uJK) / len(K)
                    dR_dbk = 2 * self.lambda_b * b_K / len(K)
                    df_dVk = df_dbk * U_u
                    dR_dVk = 2 * self.lambda_v * V_bar_K / len(K)

                    norm_nabula_Vk = np.linalg.norm((df_dVk - dR_dVk), ord=2)

                    # update V_k = V_k + gamma * (df_dVk - dR_dVk)
                    self.V[:-1, sorted(K)] += self.gamma * (df_dVk - dR_dVk)[:, None]  # trick: transpose here
                    # update b_k = b_k + gamma * (df_dbk - dR_dbk)
                    self.V[-1, sorted(K)] += self.gamma * (df_dbk - dR_dbk)
                else:
                    norm_nabula_Vk = 0

                    # calculate loss
                    f_Theta = np.log(sigmoid(r_hat_uIJ / spec_alpha_u)) + np.log(sigmoid(r_hat_uJK))
                    regula = self.lambda_u * np.linalg.norm(U_u, ord=2) + self.lambda_v * (
                        (np.linalg.norm(V_bar_I, ord=2) if len(I) != 0 else 0) + (
                            np.linalg.norm(V_bar_J, ord=2) if len(J) != 0 else 0) + (
                            np.linalg.norm(V_bar_K, ord=2)) if len(K) != 0 else 0) + self.lambda_b * (
                                     (b_I if len(I) != 0 else 0) ** 2 + (b_J if len(J) != 0 else 0) ** 2 + (
                                 b_K if len(K) != 0 else 0) ** 2)
                    bprh_loss = f_Theta - regula

                    # calculate metrics on test data
                    user_to_eval = sorted(set(self.test_data.UserID))
                    scoring_list_5, precision_5, recall_5 = self.scoring(user_to_eval=user_to_eval,
                                                                         ground_truth=self.test_data, K=5)
                    scoring_list_10, precision_10, recall_10 = self.scoring(user_to_eval=user_to_eval,
                                                                            ground_truth=self.test_data,
                                                                            K=10)

                # update estimation
                self.estimation = np.dot(self.U, self.V)
                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                t.set_postfix(loss=bprh_loss, precision_5=precision_5, recall_5=recall_5, precision_10=precision_10,
                              recall_10=recall_10, norm_nabula_U_u=norm_nabula_U_u, norm_nabula_Vi=norm_nabula_Vi,
                              norm_nabula_Vj=norm_nabula_Vj, norm_nabula_Vk=norm_nabula_Vk)
                if plot_metric:
                    plot_losses.update({
                        'Precision@5': precision_5,
                        'Precision@10': precision_10,
                        'Recall@5': recall_5,
                        'Recall@10': recall_10
                    })
                    plot_losses.send()

    def predict_estimation(self, user_to_predict, item_to_predict=None):
        if item_to_predict is None:
            return np.dot(self.U[adv_index(self.user_original_id_list, user_to_predict), :], self.V)
        else:
            return np.dot(self.U[adv_index(self.user_original_id_list, user_to_predict), :],
                          self.V[:, adv_index(self.item_original_id_list, item_to_predict)])

    def recommend(self, user_to_recommend=None, K=5):
        if self.train_data is None:
            print("Train data has not been feed")
            return None
        if user_to_recommend is None:
            user_to_recommend = self.user_list

        estimated_pref = self.predict_estimation(user_to_predict=self.user_original_id_list)

        # build the dict that user's purchased item
        user_purchased_item = dict()
        # build the rec list for users
        user_rec_list = []

        # print("Build User's Purchased Item Dict & Rec List")
        # user_set_bar = tqdm(user_to_recommend)
        for u in user_to_recommend:
            user_purchased_item[u] = set(
                self.train_data[(self.train_data.UserID == u) & (self.train_data.Action == 'P')].ItemID)
            est_pref_of_u = estimated_pref[u, :]
            #
            est_pref_sort_index = est_pref_of_u.argsort()[::-1]
            rec_item_cnt = 0
            index_cnt = 0
            while rec_item_cnt < K:
                if est_pref_sort_index[index_cnt] in user_purchased_item[u]:
                    index_cnt += 1
                else:
                    user_rec_list.append([u, est_pref_sort_index[index_cnt]])
                    index_cnt += 1
                    rec_item_cnt += 1

        self.user_purchased_item = user_purchased_item
        return pd.DataFrame(user_rec_list, columns=['UserID', 'ItemID'])

    def scoring(self, ground_truth, K=5, user_to_eval=None, y=None):
        """

        :param user_to_eval: user list to evaluate performance
        :param ground_truth: ground truth of user browsing behavior
        :param K: top K items to recommend
        :param y: ignore
        :return: precision@k
        """
        if user_to_eval is None:
            user_to_eval = sorted(set(ground_truth.UserID))

        user_to_eval = sorted(set(user_to_eval))
        # get top K recommendation list
        rec_list = self.recommend(user_to_recommend=user_to_eval, K=K)
        scoring_list = []
        # clean ground truth
        ground_truth = ground_truth[ground_truth.Action == 'P']
        # begin iteration
        for u in user_to_eval:
            rec_list_for_user_u = set(rec_list[rec_list.UserID == u].ItemID)
            ground_truth_for_user_u = set(ground_truth[ground_truth.UserID == u].ItemID)
            precision_K_for_u = len(rec_list_for_user_u.intersection(ground_truth_for_user_u)) / K
            recall_K_for_u = len(rec_list_for_user_u.intersection(ground_truth_for_user_u)) / len(
                ground_truth_for_user_u) if len(ground_truth_for_user_u) != 0 else np.nan
            scoring_list.append([u, precision_K_for_u, recall_K_for_u])
        scoring_list = pd.DataFrame(scoring_list, columns=['UserID', 'Precision@' + str(K), 'Recall@' + str(K)])
        precision_K = scoring_list.mean()['Precision@' + str(K)]
        recall_K = scoring_list.mean()['Recall@' + str(K)]

        return scoring_list, precision_K, recall_K

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha, "recursive": self.recursive}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
