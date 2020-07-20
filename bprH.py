'''
Implementation for Qiu, Huihuai, et al.
"BPRH: Bayesian personalized ranking for heterogeneous implicit feedback." Information Sciences 453 (2018): 80-98.

Author: Yihong Liu
For more details, please visit https://liu-yihong.github.io/2020/06/26/Understanding-BPR-COFISET-and-BPRH/
'''
import pickle
import numpy as np
from random import choice
from tqdm import tqdm, trange
from livelossplot import PlotLosses


def adv_index(list_to_index, list_to_match):
    return [ind for ind, match in enumerate(list_to_index) if match in list_to_match]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def indicator(z):
    return 1 if z else 0


def indicator_len(z):
    return len(z) if len(z) != 0 else 1


class bprH(object):

    def __init__(self, dim=10, omega=1, rho=1, lambda_u=0.5, lambda_v=0.1, lambda_b=0.1, gamma=0.001, num_iter=200,
                 random_state=None,
                 existed_model_path=None):
        """
        Initializing class instance bprH
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

        self.I_u_t_train = dict()
        self.I_u_a_train = dict()

        self.I_u_t_test = dict()

        self.eval_hist = []

        self.existed_model_path = existed_model_path

        if existed_model_path is not None:
            print("Loading Pre-trianed Model")
            f = open(self.existed_model_path, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            self.__dict__.update(tmp_dict)
            setattr(self, "existed_model_path", existed_model_path)
            setattr(self, "num_iter", num_iter)

    def load(self, model_path):
        f = open(model_path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self, model_path):
        f = open(model_path, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def auxiliary_target_correlation(self, X, y=None):
        """
        Calculate auxiliary-target correlation C for every user and each types of auxiliary action
        Here we only have one auxiliary action 'V' for 'View'
        :param X: train data as input
        :param y: ignore
        :return: return alpha_u
        """
        print("Calculate auxiliary-target correlation")
        target_action = 'P'
        auxiliary_action = ['V']
        alpha_u = dict()
        user_set_bar = tqdm(self.user_list)
        for u in user_set_bar:
            alpha_u[u] = dict()
            I_t_u = self.I_u_t_train[u]
            # Notice we only have View action so we do not need filtered item set
            for x in auxiliary_action:
                I_a_u = self.I_u_a_train[u]
                # Equation Reference to page 86 section 3.3
                # NOTE
                # if I_t_u is 0, then we set C_u_at to be 0
                # if I_a_u is 0, then we set C_u_ta to be 0
                C_u_at = len(I_t_u.intersection(I_a_u)) / len(I_t_u) if len(I_t_u) != 0 else 0
                C_u_ta = len(I_t_u.intersection(I_a_u)) / len(I_a_u) if len(I_a_u) != 0 else 0
                # if C_u_ta + C_u_at == 0, then we set alpha_u of user u to be 1
                # hence, C_u_X here is 1 / omega because alpha_u = omega * C_u_X
                # in this case, user u is not in train, perhaps in test
                C_u_X = 2 * C_u_at * C_u_ta / (C_u_ta + C_u_at) if C_u_ta + C_u_at != 0 else (1 / self.omega)
                # set final alpha_u to 1 if C_u_ta + C_u_at == 0
                alpha_u[u][x] = C_u_X
                # We have only one auxiliary action 'V'
                alpha_u[u]['alpha'] = self.omega * self.rho * C_u_X

        # temp = pd.DataFrame.from_dict(C_u, orient='index')
        # We have only one auxiliary action 'V'
        # temp['alpha'] = self.omega * self.rho * temp.V
        # alpha_u = temp
        # del temp
        # alpha_u.reset_index(inplace=True)
        # alpha_u.columns = ['UserID', 'V', 'alpha']
        return alpha_u

    def itemset_coselection(self, X, y=None):
        """

        :param X: trained data
        :param y: ignore
        :return: co-selection dictionary
        """

        S = dict()
        U = dict()
        # first we build U_i for each item i
        print("Generate Itemset Coselection - Build U_i")
        item_set_bar = tqdm(self.item_list)
        for i in item_set_bar:
            U[i] = set(X[(X.ItemID == i) & (X.Action == 'P')].UserID)
        del item_set_bar
        # then we build coselection dictionary
        print("Generate Itemset Coselection - Build S")
        item_set_bar = tqdm(self.item_list)
        for i in item_set_bar:
            S[i] = set()
            # NOTE: overcome the cases when U[i] = 1
            S[i].add(i)
            for j in self.item_list:
                # If more than 2 users have target action on i and j, then j is added in S[i]
                if len(U[i].intersection(U[j])) >= 2:
                    S[i].add(j)
        # save coselection list
        # with open(saved_path, 'wb') as f:
        #    pickle.dump(S, f, pickle.HIGHEST_PROTOCOL)
        return S

    def build_itemset_for_user(self):
        print("Build I_u_t, I_u_a")
        user_set_bar = tqdm(self.user_list)
        for u in user_set_bar:
            self.I_u_t_train[u] = set(
                self.train_data[(self.train_data.UserID == u) & (self.train_data.Action == 'P')].ItemID)
            self.I_u_t_test[u] = set(
                self.test_data[(self.test_data.UserID == u) & (self.test_data.Action == 'P')].ItemID)
            # here we only have one auxiliary action 'V'
            self.I_u_a_train[u] = set(self.train_data[(self.train_data.UserID == u) & (
                    self.train_data.Action == 'V')].ItemID)

    def fit(self, X, eval_X, y=None,
            model_saved_path='bprh_model.pkl',
            iter_to_save=5000,
            coselection_saved_path='data/item-set-coselection.pkl', iter_to_log=100,
            correlation=True, coselection=False, plot_metric=False, log_metric=False):
        # Here we do not load model -> train a new model
        if self.existed_model_path is None:
            # To make sure train and test works with inconsistent user and item list,
            # we transform user and item's string ID to int ID so that their ID is their index in U and V
            print("Registering Model Parameters")
            # rename user and item
            self.user_original_id_list = sorted(set(X.UserID).union(set(eval_X.UserID)))
            self.item_original_id_list = sorted(set(X.ItemID).union(set(eval_X.ItemID)))

            self.train_data = X.copy()
            self.test_data = eval_X.copy()

            self.train_data.UserID = self.train_data.UserID.apply(lambda x: self.user_original_id_list.index(x))
            self.train_data.ItemID = self.train_data.ItemID.apply(lambda x: self.item_original_id_list.index(x))

            self.test_data.UserID = self.test_data.UserID.apply(lambda x: self.user_original_id_list.index(x))
            self.test_data.ItemID = self.test_data.ItemID.apply(lambda x: self.item_original_id_list.index(x))

            self.item_list = [idx[0] for idx in enumerate(self.item_original_id_list)]
            self.user_list = [idx[0] for idx in enumerate(self.user_original_id_list)]

            self.num_u = len(self.user_list)
            self.num_i = len(self.item_list)

            # build I_u_t, I_u_a (pre-computing for acceleration)
            self.build_itemset_for_user()

            # Calculate auxiliary-target correlation C for every user and each types of auxiliary action
            if correlation:
                self.alpha_u = self.auxiliary_target_correlation(X=self.train_data)
            else:
                print("No auxiliary-target correlation - all alpha_u equal to one")
                alpha_u_all_ones = dict()
                user_set_bar = tqdm(self.user_list)
                for u in user_set_bar:
                    alpha_u_all_ones[u] = dict()
                    alpha_u_all_ones[u]['alpha'] = 1.0
                self.alpha_u = alpha_u_all_ones.copy()

            # Generate item-set based on co-selection
            if coselection:
                self.S = self.itemset_coselection(X=self.train_data)

            # Initialization of User and Item Matrices
            if self.random_state is not None:
                np.random.seed(self.random_state)
            else:
                np.random.seed(0)

            print("Initializing User and Item Matrices")
            # NOTE: Initialization is influenced by mean and std
            self.U = np.random.normal(size=(self.num_u, self.dim + 1), loc=0.0, scale=0.1)
            self.V = np.random.normal(size=(self.dim + 1, self.num_i), loc=0.0, scale=0.1)
            # self.U = cupy.zeros(shape=(self.num_u, self.dim + 1))
            # self.V = cupy.zeros(shape=(self.dim + 1, self.num_i))
            self.U[:, -1] = 1.0
            # estimation is U dot V
            self.estimation = np.dot(self.U, self.V)

        # Configure loss plots layout
        if plot_metric:
            groups = {'Precision@K': ['Precision@5', 'Precision@10'],
                      'Recall@K': ['Recall@5', 'Recall@10'],
                      'AUC': ['AUC']}
            plot_losses = PlotLosses(groups=groups)

        # Start Iteration
        all_item = set(self.item_list)
        user_in_train = sorted(set(self.train_data.UserID))
        print("Start Training")
        with trange(self.num_iter) as t:
            for index in t:
                # Description will be displayed on the left
                # t.set_description('ITER %i' % index)

                # Build u, I, J, K
                # uniformly sample a user from U
                u = choice(user_in_train)

                # build I
                # uniformly sample a item i from I_u_t
                I_u_t = self.I_u_t_train[u]
                if len(I_u_t) != 0:
                    i = choice(sorted(I_u_t))
                    # build I = I_u_t cap S_i
                    if coselection:
                        I = I_u_t.intersection(self.S[i])
                    else:
                        # if no coselection, we set I as the set of purchased items by user u
                        # no uniform sampling, like COFISET
                        I = I_u_t
                else:  # if no item in I_u_t, then set I to empty set
                    i = None
                    I = set()

                # build J, since we only have one auxiliary action, we follow the uniform sampling
                I_u_oa = self.I_u_a_train[u] - I_u_t
                if len(I_u_oa) != 0:
                    j = choice(sorted(I_u_oa))
                    if coselection:
                        # NOTE: typo in paper?
                        J = I_u_oa.intersection(self.S[j])
                    else:
                        # if no coselection, we set J as the set of only-auxiliary items by user u
                        # no uniform sampling, like COFISET
                        J = I_u_oa
                else:  # if no item in I_u_oa, then set J to empty set
                    j = None
                    J = set()

                # build K
                I_u_n = all_item - I_u_t - I_u_oa
                if len(I_u_n) != 0:
                    k = choice(sorted(I_u_n))
                    # build K
                    if coselection:
                        # NOTE: typo in paper?
                        K = I_u_n.intersection(self.S[k])
                    else:
                        # if no coselection, we set K as the set of no-action items by user u
                        # no uniform sampling, like COFISET
                        K = I_u_n
                else:  # if no item in I_u_n, then set K to empty set
                    k = None
                    K = set()

                # calculate intermediate variables
                # get specific alpha_u
                spec_alpha_u = self.alpha_u[u]['alpha']

                U_u = self.U[u, :-1].copy()
                # get r_hat_uIJ, r_hat_uJK, r_hat_uIK
                r_hat_uI = np.average(self.estimation[u, sorted(I)]) if len(
                    I) != 0 else np.array([0])
                r_hat_uJ = np.average(self.estimation[u, sorted(J)]) if len(
                    J) != 0 else np.array([0])
                r_hat_uK = np.average(self.estimation[u, sorted(K)]) if len(
                    K) != 0 else np.array([0])

                r_hat_uIJ = r_hat_uI - r_hat_uJ
                r_hat_uJK = r_hat_uJ - r_hat_uK
                r_hat_uIK = r_hat_uI - r_hat_uK
                # get V_bar_I, V_bar_J, V_bar_K
                V_bar_I = np.average(self.V[:-1, sorted(I)], axis=1) if len(
                    I) != 0 else np.zeros(
                    shape=(self.dim,))
                V_bar_J = np.average(self.V[:-1, sorted(J)], axis=1) if len(
                    J) != 0 else np.zeros(
                    shape=(self.dim,))
                V_bar_K = np.average(self.V[:-1, sorted(K)], axis=1) if len(
                    K) != 0 else np.zeros(
                    shape=(self.dim,))
                # get b_I, b_J, b_K
                b_I = np.average(self.V[-1, sorted(I)]) if len(I) != 0 else np.array([0])
                b_J = np.average(self.V[-1, sorted(J)]) if len(J) != 0 else np.array([0])
                b_K = np.average(self.V[-1, sorted(K)]) if len(K) != 0 else np.array([0])

                # here we want to examine the condition of empty sets
                indicator_I = indicator(len(I) == 0)
                indicator_J = indicator(len(J) == 0)
                indicator_K = indicator(len(K) == 0)
                indicator_sum = indicator_I + indicator_J + indicator_K

                if 0 <= indicator_sum <= 1:
                    # these are the cases when only one set are empty or no set is empty
                    # when all three are not empty, or I is empty, or K is empty, it is
                    # easy to rewrite the obj by multiplying the indicator
                    # when J is empty, we have to rewrite the obj
                    if indicator_J == 1:
                        # when J is empty

                        # NABLA U_u
                        df_dUu = sigmoid(- r_hat_uIK) * (V_bar_I - V_bar_K)
                        dR_dUu = 2 * self.lambda_u * U_u
                        # update U_u = U_u + gamma * (df_dUu - dR_dUu)
                        self.U[u, :-1] += self.gamma * (df_dUu - dR_dUu)

                        # NABLA V_i
                        df_dbi = (1 - indicator_I) * sigmoid(- r_hat_uIK) / indicator_len(I)
                        dR_dbi = (1 - indicator_I) * 2 * self.lambda_b * b_I / indicator_len(I)
                        df_dVi = df_dbi * U_u
                        dR_dVi = 2 * self.lambda_v * V_bar_I / indicator_len(I)
                        # update V_i = V_i + gamma * (df_dVi - dR_dVi)
                        self.V[:-1, sorted(I)] += self.gamma * (df_dVi - dR_dVi)[:, None]  # trick: transpose here
                        # update b_i = b_i + gamma * (df_dbi - dR_dbi)
                        self.V[-1, sorted(I)] += self.gamma * (df_dbi - dR_dbi)

                        # No change on J

                        # NABLA V_k
                        df_dbk = (1 - indicator_K) * - sigmoid(- r_hat_uIK) / indicator_len(K)
                        dR_dbk = (1 - indicator_K) * 2 * self.lambda_b * b_K / indicator_len(K)
                        df_dVk = df_dbk * U_u
                        dR_dVk = 2 * self.lambda_v * V_bar_K / indicator_len(K)

                        # update V_k = V_k + gamma * (df_dVk - dR_dVk)
                        self.V[:-1, sorted(K)] += self.gamma * (df_dVk - dR_dVk)[:, None]  # trick: transpose here
                        # update b_k = b_k + gamma * (df_dbk - dR_dbk)
                        self.V[-1, sorted(K)] += self.gamma * (df_dbk - dR_dbk)

                    else:
                        # when J is not empty
                        # NABLA U_u
                        df_dUu = (1 - indicator_I) * sigmoid(- r_hat_uIJ / spec_alpha_u) / spec_alpha_u * (
                                V_bar_I - V_bar_J) + \
                                 (1 - indicator_K) * sigmoid(- r_hat_uJK) * (V_bar_J - V_bar_K)
                        dR_dUu = 2 * self.lambda_u * U_u
                        # update U_u = U_u + gamma * (df_dUu - dR_dUu)
                        self.U[u, :-1] += self.gamma * (df_dUu - dR_dUu)

                        # NABLA V_i
                        df_dbi = (1 - indicator_I) * sigmoid(- r_hat_uIJ / spec_alpha_u) / (
                                indicator_len(I) * spec_alpha_u)
                        dR_dbi = (1 - indicator_I) * 2 * self.lambda_b * b_I / indicator_len(I)
                        df_dVi = df_dbi * U_u
                        dR_dVi = 2 * self.lambda_v * V_bar_I / indicator_len(I)
                        # update V_i = V_i + gamma * (df_dVi - dR_dVi)
                        self.V[:-1, sorted(I)] += self.gamma * (df_dVi - dR_dVi)[:, None]  # trick: transpose here
                        # update b_i = b_i + gamma * (df_dbi - dR_dbi)
                        self.V[-1, sorted(I)] += self.gamma * (df_dbi - dR_dbi)

                        # NABLA V_j
                        df_dbj = (1 - indicator_I) * (- sigmoid(- r_hat_uIJ / spec_alpha_u) / spec_alpha_u +
                                                      (1 - indicator_K) * sigmoid(- r_hat_uJK)) / indicator_len(J)
                        dR_dbj = 2 * self.lambda_b * b_J / indicator_len(J)
                        df_dVj = df_dbj * U_u
                        dR_dVj = 2 * self.lambda_v * V_bar_J / indicator_len(J)

                        # update V_j = V_j + gamma * (df_dVj - dR_dVj)
                        self.V[:-1, sorted(J)] += self.gamma * (df_dVj - dR_dVj)[:, None]  # trick: transpose here
                        # update b_j = b_j + gamma * (df_dbj - dR_dbj)
                        self.V[-1, sorted(J)] += self.gamma * (df_dbj - dR_dbj)

                        # NABLA V_k
                        df_dbk = (1 - indicator_K) * - sigmoid(- r_hat_uJK) / indicator_len(K)
                        dR_dbk = (1 - indicator_K) * 2 * self.lambda_b * b_K / indicator_len(K)
                        df_dVk = df_dbk * U_u
                        dR_dVk = 2 * self.lambda_v * V_bar_K / indicator_len(K)

                        # update V_k = V_k + gamma * (df_dVk - dR_dVk)
                        self.V[:-1, sorted(K)] += self.gamma * (df_dVk - dR_dVk)[:, None]  # trick: transpose here
                        # update b_k = b_k + gamma * (df_dbk - dR_dbk)
                        self.V[-1, sorted(K)] += self.gamma * (df_dbk - dR_dbk)

                else:
                    # these are the cases when at least two sets are empty
                    # at these cases, we ignore this user and continue the loop
                    continue

                # calculate loss
                # f_Theta = np.log(sigmoid(r_hat_uIJ / spec_alpha_u)) + np.log(sigmoid(r_hat_uJK))
                # regula = self.lambda_u * np.linalg.norm(U_u, ord=2) + self.lambda_v * (
                #        (np.linalg.norm(V_bar_I, ord=2) if len(I) != 0 else 0) + (
                #            np.linalg.norm(V_bar_J, ord=2) if len(J) != 0 else 0) + (
                #            np.linalg.norm(V_bar_K, ord=2)) if len(K) != 0 else 0) + self.lambda_b * (
                #                     (b_I if len(I) != 0 else 0) ** 2 + (b_J if len(J) != 0 else 0) ** 2 + (
                #                 b_K if len(K) != 0 else 0) ** 2)
                # bprh_loss = f_Theta - regula

                # update estimation
                old_estimation = self.estimation.copy()
                self.estimation = np.dot(self.U, self.V)
                # estimation changed
                est_changed = np.linalg.norm(self.estimation - old_estimation)

                # we only save model to file when the num of iter % iter_to_save == 0
                if (index + 1) % iter_to_save == 0:
                    self.save(model_path=model_saved_path + "_" + str(index))

                # we only calculate metric when the num of iter % iter_to_log == 0
                if (index + 1) % iter_to_log == 0:
                    if log_metric | plot_metric:
                        # calculate metrics on test data
                        user_to_eval = sorted(set(self.test_data.UserID))
                        scoring_list_5, precision_5, recall_5, avg_auc = self.scoring(user_to_eval=user_to_eval,
                                                                                      ground_truth=self.test_data,
                                                                                      K=5,
                                                                                      train_data_as_reference_flag=True)
                        scoring_list_10, precision_10, recall_10, _ = self.scoring(user_to_eval=user_to_eval,
                                                                                   ground_truth=self.test_data,
                                                                                   K=10,
                                                                                   train_data_as_reference_flag=True)
                    if log_metric:
                        self.eval_hist.append([index, precision_5, precision_10, recall_5, recall_10, avg_auc])

                    if plot_metric:
                        plot_losses.update({
                            'Precision@5': precision_5,
                            'Precision@10': precision_10,
                            'Recall@5': recall_5,
                            'Recall@10': recall_10,
                            'AUC': avg_auc
                        })
                        plot_losses.send()

                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                t.set_postfix(
                    est_changed=est_changed,
                    len_I=len(I),
                    len_J=len(J),
                    len_K=len(K))

    def predict_estimation(self, user_to_predict, item_to_predict=None):
        if item_to_predict is None:
            return np.dot(self.U[adv_index(self.user_original_id_list, user_to_predict), :], self.V)
        else:
            return np.dot(self.U[adv_index(self.user_original_id_list, user_to_predict), :],
                            self.V[:, adv_index(self.item_original_id_list, item_to_predict)])

    def recommend(self, user_to_recommend=None, K=5,
                  train_data_as_reference_flag=True,
                  ignore_user_not_in_train=False):
        if self.train_data is None:
            print("Train data has not been feed")
            return None
        if user_to_recommend is None:
            user_to_recommend = self.user_list

        user_in_train = set(self.train_data.UserID)
        if not ignore_user_not_in_train:
            # In order to address the case when a user is in test but not in train
            # we build Popularity based ranking

            ranking_list = self.train_data.groupby("ItemID").count().UserID.copy()
            ranking_list.sort_values(inplace=True, ascending=False)
            ranking_list = ranking_list.index.to_list()

        # build the rec list for users
        user_rec_dict = dict()

        for u in user_to_recommend:

            # what if user u is in test data but not in train?
            if u not in user_in_train:
                if ignore_user_not_in_train:
                    # if we ignore this user, then we skip it
                    continue
                else:
                    # otherwise, we use Popularity based ranking for this user
                    if u not in user_in_train:
                        user_rec_dict[u] = set(ranking_list[:K])
                        continue

            est_pref_of_u = self.estimation[u, :]
            # Next is the case when user u is in train data
            # get the ranking for user u's pref of item
            user_rec_dict[u] = set()
            est_pref_sort_index = est_pref_of_u.argsort()[::-1].get()
            rec_item_cnt = 0
            # case of recommending on test data
            if train_data_as_reference_flag:
                for item_id in est_pref_sort_index:
                    if rec_item_cnt == K:
                        break
                    # we only consider the item that is not in train data for user u
                    if item_id not in self.I_u_t_train[u]:
                        user_rec_dict[u].add(item_id)
                        rec_item_cnt += 1
            # case of recommending on train data
            else:
                user_rec_dict[u] = set(est_pref_sort_index[:K])

        return user_rec_dict

    def scoring(self, ground_truth, K=5,
                user_to_eval=None, y=None,
                train_data_as_reference_flag=True,
                ignore_user_not_in_train=False,
                use_min_of_K_and_size_of_groundtruth=False
                ):
        """

        :param train_data_as_reference_flag:
        :param ignore_user_not_in_train:
        :param use_min_of_K_and_size_of_groundtruth:
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
        user_rec_dict = self.recommend(user_to_recommend=user_to_eval,
                                       K=K,
                                       train_data_as_reference_flag=train_data_as_reference_flag,
                                       ignore_user_not_in_train=ignore_user_not_in_train)
        scoring_list = []
        # clean ground truth
        ground_truth_cleaned = ground_truth[ground_truth.Action == 'P']
        # for AUC calculation under train or test
        I_u_t_in_auc = None
        # build two sets with users for AUC
        #
        # what if grount_truth comes from train
        if train_data_as_reference_flag:
            if len(self.I_u_t_test) == 0:
                for u in user_to_eval:
                    self.I_u_t_test[u] = set(ground_truth_cleaned[ground_truth_cleaned.UserID == u].ItemID)
            I_u_t_in_auc = self.I_u_t_test
        else:
            I_u_t_in_auc = self.I_u_t_train
        # begin iteration
        for u in user_to_eval:
            u_in_train_or_not = (u in user_rec_dict.keys())
            if ignore_user_not_in_train & (not u_in_train_or_not):
                # CASE: user not in train and we ignore it
                continue
            # Otherwise, we continue
            rec_list_for_user_u = user_rec_dict[u]
            # get precision and recall
            I_u_t = set(ground_truth_cleaned[ground_truth_cleaned.UserID == u].ItemID)
            # what if the ground truth size for user u is smaller than K
            if use_min_of_K_and_size_of_groundtruth:
                precision_K_for_u = len(rec_list_for_user_u.intersection(I_u_t)) / min(K, len(I_u_t))
            else:
                precision_K_for_u = len(rec_list_for_user_u.intersection(I_u_t)) / K

            recall_K_for_u = len(rec_list_for_user_u.intersection(I_u_t)) / len(
                I_u_t) if len(I_u_t) != 0 else np.nan
            # get auc
            est_pref_of_u = self.estimation[u, :].get()
            E_u = 0
            indicator_cnt = 0
            for i in I_u_t_in_auc[u]:
                for j in set(self.item_list) - I_u_t_in_auc[u].union(self.I_u_t_train[u]):
                    E_u += 1
                    if est_pref_of_u[i] > est_pref_of_u[j]:
                        indicator_cnt += 1
            auc_for_u = indicator_cnt / E_u if E_u != 0 else 0
            scoring_list.append([u, precision_K_for_u, recall_K_for_u, auc_for_u])
        # scoring_list = pd.DataFrame(scoring_list, columns=['UserID', 'Precision@' + str(K), 'Recall@' + str(K)])
        # precision_K = scoring_list.mean()['Precision@' + str(K)]
        # recall_K = scoring_list.mean()['Recall@' + str(K)]
        scoring_list = np.array(scoring_list)
        scoring_average = np.nanmean(scoring_list, axis=0)
        precision_K = scoring_average[1]
        recall_K = scoring_average[2]
        avg_auc = scoring_average[3]
        return scoring_list, precision_K, recall_K, avg_auc

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha, "recursive": self.recursive}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
