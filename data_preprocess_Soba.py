import pandas as pd
import math

# According to TBPR paper, Soba dataset includes
# 4,712 users perform 18,267 purchase, 225,651 view and 100,067
# like over 7,015 items
# Why they exclude users without any purchase?
# Why they exclude items without any purchase?
soba = pd.read_csv("data/Sobazaar-hashID.csv")
action_set = set(soba.Action)
# Description of Actions
#  'product_detail_clicked' -> auxiliary behavior (view)
#  'product_wanted' -> auxiliary behavior (like)
#  'purchase:buy_clicked' -> purchase behavior (target)
#  'content:interact:product_clicked',
#  'content:interact:product_detail_viewed',
#  'content:interact:product_wanted',
#  'pixel-init',
#  'pixel-order',
#  'pixel-order-no-reference',
item_set = set(soba.ItemID)
user_set = set(soba.UserID)

# Users with at least one purchase behavior
user_p_id = set(soba[soba.Action == 'purchase:buy_clicked'].UserID)
temp = soba[soba.UserID.isin(user_p_id)]
temp = temp[temp.Action.isin(['product_detail_clicked', 'product_wanted', 'purchase:buy_clicked'])]
soba_cleaned_1 = temp.sort_values(by=['UserID', 'Timestamp'])
del temp
soba_cleaned_1.reset_index(drop=True, inplace=True)

# Users with at least one purchase &
# Items with at least one purchase
item_p_id = set(soba_cleaned_1[soba_cleaned_1.Action == 'purchase:buy_clicked'].ItemID)
soba_cleaned_2 = soba_cleaned_1[soba_cleaned_1.ItemID.isin(item_p_id)]
soba_cleaned_2.reset_index(drop=True, inplace=True)

# For a user u, find the item-set group (I,J,K)
soba_cleaned = soba_cleaned_2
for u in user_p_id:
    I_u = set(soba_cleaned[soba_cleaned.UserID == u][soba_cleaned.Action == 'purchase:buy_clicked'].ItemID)
    J_u = set(soba_cleaned[soba_cleaned_1.UserID == u][
                  soba_cleaned.Action.isin(['product_wanted', 'product_detail_clicked'])].ItemID) - I_u
    # Specify the item set
    K_u = item_p_id - I_u - J_u
    print(u, len(I_u), len(J_u), len(K_u))

# Find user u's auxiliary-target correlation based on co-occurrence
soba_cleaned = soba_cleaned_2
for u in user_p_id:
    I_t = set(soba_cleaned[(soba_cleaned.UserID == u) &
                           (soba_cleaned.Action == 'purchase:buy_clicked')].ItemID)
    I_a_view = set(soba_cleaned[(soba_cleaned.UserID == u) &
                                (soba_cleaned.Action == 'product_detail_clicked')].ItemID)
    I_a_like = set(soba_cleaned[(soba_cleaned.UserID == u) &
                                (soba_cleaned.Action == 'product_wanted')].ItemID)
    