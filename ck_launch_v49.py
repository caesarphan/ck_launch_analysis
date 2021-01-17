# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:16:19 2020

@author: caesa
"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
%matplotlib inline
# from datetime import timedelta

events = pd.read_csv('cryptokitties_data_11112020.csv')
users = pd.read_csv('cryptokitties_user_attributes_11112020.csv')
cats = pd.read_csv('cryptokitty_attributes_11112020.csv')

received_promotion = pd.read_csv('beta_test_users_receiving_cat_promotion.csv')
received_promotion = received_promotion.rename(columns = {'value':'sender_id'})
received_promotion = received_promotion.sender_id.str.lower()

beta_group = ['2017-11-23','2017-11-24','2017-11-25','2017-11-26','2017-11-27']
save = events

# events = events[events.ymd < '2017-12-13']
# events = events[(events.event == 'bid') & (events.status == 1)]



# =============================================================================
# events.columns Index(['transaction_id', 'ymdhms', 'kitty_id', 'event', 'sender_id',
#        'smart_contract', 'recipient_id', 'status', 'nonce', 'starting_price',
#        'ending_price', 'duration', 'price', 'seller_id', 'seller_axiom',
#        'time_on_market', 'ymd', 'eth_usd_exchange_rate', 'price_USD',
#        'gas_price_eth', 'gas_used_eth', 'gas_limit_eth', 'gas_used_USD',
#        'gas_limit_USD', 'gas_used_percent', 'breeding_fee_eth',
#        'breeding_fee_USD', 'generation'],
#       dtype='object')
# =============================================================================

# KPI To Calculate
## Daily sales over time - Yes
## Total customers acquired - Yes
## Customer acquisition cost (CAC) - Yes
## Distribution of spend per purchase
## Initial versus repeat sales volume
## Initial versus repeat average order value (AOV)
## Sales and AOV by source
## First-purchase profitability
## Cohorted sales (the “C3”)
## Revenue retention curves
## Cumulative spend per customer
## Distribution of total spend by customer
## Customer concentration (“Pareto”) chart- Yes


#User by Cohort
user_attr_cols = ['user_id','cohort_sending_action']
user_attributes = users[user_attr_cols]
user_attributes = user_attributes[user_attributes['cohort_sending_action'].notna()][user_attr_cols]

user_attributes = user_attributes.rename({'user_id':'sender_id'})


#group the lowest join date of each sender
join_date = events.groupby('sender_id')['ymd'].min()

# events = save

#Run only once
events = pd.merge(events, join_date, on = 'sender_id', how = 'left')
events.rename(columns= {'ymd_y':'join_date'}, inplace = True)
# events['join_date'] = events['join_date'].astype('datetime64[ns]')
events['join_date'] = pd.to_datetime(events['join_date']).dt.date

#use only once to identify cohort and join date to Key-Value pair
temp = events['join_date'].sort_values().unique()
temp_cohort = np.array((range(1,len(temp)+1)))
cohort_key = dict(zip(temp, temp_cohort))


events.rename(columns = {'ymd_x':'ymd'}, inplace = True)
events['cohort'] = events.join_date.apply(lambda x: cohort_key[x])

del [temp, temp_cohort, cohort_key]

#change data type to date to find cohort activity date
events['ymd'] = pd.to_datetime(events['ymd'], format = '%Y-%m-%d').dt.date

events["c_days"] = (events['ymd'] - events['join_date']).dt.days

# events['ymd']
# events["c_days"] = events["c_days"].dt.days

# cohort_activity = events.pivot(index = 'join_date', columns = 'c_days', aggfunc = 'count', fill_value = 0)

# cohort_dau = events.groupby(['join_date','c_days'])['sender_id'].nunique().unstack().reset_index()
l_cohort_dau = events.groupby(['join_date','cohort','c_days'])['sender_id'].nunique().reset_index()
l_cohort_dau.rename(columns ={'sender_id':'act_users'}, inplace = True)


l_cohort_cop = l_cohort_dau.copy()
del l_cohort_cop['cohort']
w_cohort_dau = l_cohort_cop.pivot_table(index = ['join_date'],columns = 'c_days')












# temp_l_cohort_dau = events.groupby(['join_date','c_days'])['sender_id'].nunique().reset_index()
# temp_w_cohort_dau = temp_l_cohort_dau[['join_date','c_days']].pivot_table(index = 'join_date', columns = 'c_days', aggfunc = 'sum')

#Why did this code Fail????
# TEMP_w_cohort_dau = events.pivot_table(index ='join_date',columns = 'c_days')['sender_id'].nunique().unstack()

data = l_cohort_dau.copy()
data['join_date'] = data['join_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
cohort_size = data[data['c_days'] == 0].reset_index(drop=True)


data['cohort_size'] = data['join_date'].apply(lambda x: cohort_size.loc[cohort_size['join_date'] == x,'act_users'].values[0])
data['cohort_act'] = data.apply(lambda x: x['act_users'] / x['cohort_size'], axis = 1)

#daily cohort retention rate
data['RR'] = (data['act_users'] / data['cohort_size'])
data['churn'] = 1 - (data['act_users'] / data['cohort_size'])
data.sort_values(by = ['join_date','cohort','c_days'], inplace = True)

###Revenue Metrics
event_rev = ['bid','bidonsiringauction']
relevant_columns = ['ymd', 'cohort', 'age_by_day', 'direct_sales_revenue','secondary_sales_rev','cat_breeding_rev','total_revenue']
layer_cake = events.loc[(events.status==1) & (events["event"].isin(event_rev))].reset_index(drop = True)
layer_cake['join_date'] = layer_cake['join_date'].apply(lambda x: x.strftime('%Y-%m-%d'))

del [event_rev, relevant_columns]


#function to generate new columns of rev
def applicable_rev(x, axiom, event_type, rate = 1):
    output = round(x['price_USD'] * rate,2) if (x['seller_axiom'] == axiom and x['event'] == event_type) else 0

    return output

#test function works
    
layer_cake['direct_sales_rev'] = layer_cake.apply(applicable_rev, args = (1,'bid') , axis = 1)
layer_cake['second_sales_rev'] = layer_cake.apply(applicable_rev, args = (0,'bid', 0.0375) , axis = 1)
layer_cake['breeding_rev'] = layer_cake.apply(lambda x: round(x['price_USD']*0.0375,2) if x['event'] == 'bidonsiringauction' else 0,axis = 1)
layer_cake.loc[:,'total_rev'] = layer_cake.loc[:,['direct_sales_rev','second_sales_rev','breeding_rev']].sum(axis=1, min_count=1)


rel_group = ['join_date','cohort','c_days']
rel_rev_attr = ['direct_sales_rev','second_sales_rev','breeding_rev','total_rev']
rev_data = layer_cake.groupby(rel_group)[rel_rev_attr].sum().reset_index()
rev_data = rev_data.sort_values(by = ['join_date','c_days'])



# rev_data.loc[rev_data['transaction_id']== 
#              '0x6f155dfeaf5ba62294d4230a3497170fa2062797ae8a8b09db609d31398e9eb9','total_rev']
# rev_data.loc[rev_data['transaction_id']== 
#              '0x6f155dfeaf5ba62294d4230a3497170fa2062797ae8a8b09db609d31398e9eb9','direct_sales_rev']
# rev_data.loc[rev_data['transaction_id']== 
#              '0x6f155dfeaf5ba62294d4230a3497170fa2062797ae8a8b09db609d31398e9eb9','second_sales_rev']


#Why did this code fail???
# rev_data['total_rev'] = (layer_cake['direct_sales_rev']+ layer_cake['second_sales_rev'] + layer_cake['breeding_rev']



    ###Trying to find ARPU
    
    # layer_cake_total_rev = rev_data.pivot_table(index ='join_date',columns = 'c_days', values = 'total_rev', aggfunc = 'sum')
    # # l_layer_cake = layer_cake.groupby('join_date')
    # avg_user_rev = pd.DataFrame().reindex_like(layer_cake_total_rev)
    
    # for nrow in range(len(avg_user_rev)-1):
    #     for ncol in range(nrow):
    #         avg_user_rev.iloc[nrow,ncol] = layer_cake_total_rev.iloc[nrow,ncol] / w_cohort_dau.iloc[nrow,ncol]
    
    # temp_2 = w_cohort_dau.melt()
    
    # TEMP_w_cohort_dau = w_cohort_dau.copy()
    # TEMP_w_cohort_dau.reset_index().unstack().reset_index(drop = True, inplace = True)
    # TEMP_w_cohort_dau = TEMP_w_cohort_dau.columns.name = None
    # del TEMP_w_cohort_dau.columns.name
    
    
    # range(avg_user_rev))
    
    # layer_cake_total_rev.shape[0]
    # layer_cake_total_rev.shape[1]
    
    # length = 0
    # for nrow in range(len(cohort_size)):
    #     # length +=1
    #     # print(length)
    #     for ncol in range(nrow-1):
    #         print(cohort_size.iloc[nrow, ncol])
    #         length += 1
    #         print(length)


# len(cohort_size)

#l_cohort_dau only
left_only = data.merge(rev_data, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']

#cohort_data
right_only = rev_data.merge(data, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']

#total - Must make string dates match data type
cohort_data = data.merge(rev_data, how = 'outer', left_on = ['join_date','c_days'], right_on = ['join_date','c_days'])
cohort_data['ARPU'] = cohort_data.apply(lambda x: x['total_rev'] / x['act_users'], axis = 1)

    # #BAR plot - Cohort Avg Churn
    # f, ax = plt.subplots(figsize=(14, 11))
    # plt.figure(figsize = (15,10))
    # plt.xticks(rotation= 45)
    # plt.xlabel('Cohort by Date')
    # plt.ylabel('Churn Rate')
    # plt.title('Cohort Churn Rate')
    # ax = sns.barplot(x=cohort_data['join_date'], y = cohort_data['churn'])
    
    
    # f.savefig('cohort_churn.png', dpi=1200)
    
    # #Cohort heat map
    
    # # l_cohort_dau.to_csv(r'C:\Users\caesa\Documents\UCI\Fall 2020\BANA 205\week_7\cohort_dau.csv', index = False)
    # # rev_data.to_csv(r'C:\Users\caesa\Documents\UCI\Fall 2020\BANA 205\week_7\cohort_rev.csv', index = False)
    # cohort_data.to_csv(r'C:\Users\caesa\Documents\UCI\Fall 2020\BANA 205\week_7\cohort_data.csv', index = False)
    
    # del l_cohort_dau['cohort']
    
    # # l_cohort_dau = l_cohort_dau.pivot('join_date','c_days','act_users')
    # # sns.set(style='white')
    
    # f, ax = plt.subplots(figsize=(14, 11))
    # # fig.set_size_inches(11.7, 8.27)
    # plt.title('Cohorts: User Retention')
    # temp_w_cohort_dau = l_cohort_cop.pivot('join_date','c_days','act_users')
    # ####temp2_col_average = temp_w_cohort_dau.apply(lambda x: x.mean(), axis = 0)
    
    
    # sns.set(font_scale=1)
    # ax = sns.heatmap(temp_w_cohort_dau, annot = True,annot_kws={"size": 12},fmt="0.1f")
    
    # f.savefig('cohort_activity_2.png', dpi=1200)
    
    # ###Cohort Retention Heat Map
    # f, ax = plt.subplots(figsize=(14, 11))
    # plt.title('Cohorts: User Retention')
    # temp_cohort_rr = cohort_data.pivot('join_date','c_days','RR')
    
    # sns.set(font_scale=0.8)
    # ax = sns.heatmap(temp_cohort_rr, annot = True,annot_kws={"size": 11},fmt="0.2f")
    
    # f.savefig('Cohort_retention.png', dpi=1200)
    
    
    # ###NEw Map
    # f, ax = plt.subplots(figsize=(14, 11))
    # plt.title('Cohorts: User Retention')
    # temp_arpu = cohort_data.pivot('join_date','c_days','ARPU')
    
    # sns.set(font_scale=0.8)
    # ax = sns.heatmap(temp_arpu, annot = True,annot_kws={"size": 11},fmt="0.2f",cmap = 'coolwarm')
    
    # f.savefig('APRU.png', dpi=1200)
    
    
    # sns.displot(data, x="c_days", hue="RR")


#Find number of bought/sold transactions
# l_cohort_transaction = events.groupby(['join_date','cohort','c_days'])['transaction_id'].nunique().reset_index()
# l_cohort_transaction = events.groupby(['join_date','cohort','c_days'])['transaction_id'].

l_transaction = events.loc[events['status'] == 1].reset_index(drop=True)

def applicable_tran(x, axiom, event_type, rate = 1):
    output = 1 if (x['seller_axiom'] == axiom and x['event'] == event_type and x['status'] ==1) else 0

    return output

#test function works
    
l_transaction['direct_sales_rev'] = l_transaction.apply(applicable_tran, args = (1,'bid') , axis = 1)
l_transaction['second_sales_rev'] = l_transaction.apply(applicable_tran, args = (0,'bid', 0.0375) , axis = 1)
l_transaction['breeding_rev'] = l_transaction.apply(lambda x: 1 if x['event'] == 'bidonsiringauction' else 0,axis = 1)
l_transaction.loc[:,'total_rev'] = l_transaction.loc[:,['direct_sales_rev','second_sales_rev','breeding_rev']].sum(axis=1, min_count=1)

exp_columns = ['direct_sales_rev','second_sales_rev','breeding_rev','total_rev']
exp_transaction = l_transaction.groupby(['join_date','c_days'])[exp_columns].sum().reset_index()
exp_transaction.to_csv(r'C:\Users\caesa\Documents\UCI\Fall 2020\BANA 205\week_7\transactions.csv', index = False)



#Mean spend on beta launch

events['ymd'] = pd.to_datetime(events.ymd, format='%Y-%m-%d')

beta_test_cat_price = events.loc[(events['event'] == 'bid') & (events['ymd'] < '2017-11-28') & (events['status'] == 1)]['price_USD'].mean()
cummulative_spend_on_promotions = beta_test_cat_price  * 784
# cac = cummulative_spend_on_promotions 

axiom_id = ['0xa1e12defa6dbc8e900a6596083322946c03f01e3','0xa21037849678af57f9865c6b9887f4e339f6377a',
            '0xaf1e54b359b0897133f437fc961dd16f20c045e1','0x2041bb7d8b49f0bde3aa1fa7fb506ac6c539394c','axiom']

am = events[events['sender_id'].isin(users['user_id'])][['sender_id','ymd']]


#CAC data Frame
cohort_data['cumm_spend'] = cummulative_spend_on_promotions

###Need to back out axiom from cac
cac_df = cohort_data[(cohort_data['c_days']==0)].reset_index(drop=True)
cac_df['cum_active'] = cac_df.act_users.cumsum(skipna=False)
cac_df['cac'] = cohort_data['cumm_spend']/cac_df['cum_active']

cac_df.to_csv(r'C:\Users\caesa\Documents\UCI\Fall 2020\BANA 205\week_7\cac_cohort.csv', index = False)


power_users = events.loc[(events['ymd'] < '2017-12-13') & (events['event'] == 'bidonsiringauction') & (events['status'] ==1)]
power_user_group = power_users.groupby('recipient_id')['ymd'].nunique()
# plt.hist(power_user_group, bins = power_users['join_date'].nunique()-1)
plt.title('Power User Curve')
plt.xlabel('Days Active')
plt.ylabel('Users')

plt.hist(power_user_group, bins = 21)
plt.show()

print(l_cohort_dau['join_date'].nunique())

sns.set_color_codes()
ax = sns.distplot(power_user_group, color="y")
ax.set(xlabel="Days Active", ylabel = "% Active Users Users")
fig, axes = plt.subplots(nrows=2, ncols=2)

###failed listings
rel_list_data = ['join_date','cohort','gas_used_USD','generation', 'kitty_id','status']
total_listings = events[(events['event'] == 'createsaleauction') & (events['sender_id'].isin(users.user_id))][rel_list_data]
#total number of listings in this period
total_listings['sum_lists'] = len(total_listings.iloc[:,1])

#identify failed listing each day
# failed_daily = total_listings.groupby(['join_date','status'])['cohort'].count()
failed_daily = total_listings.groupby(['join_date','status'])['cohort'].count().reset_index()
failed_daily.to_csv(r'C:\Users\caesa\Documents\UCI\Fall 2020\BANA 205\week_7\failed_transactions.csv', index = False)

kitties_gen_sold = events[(events['generation']==0) & (events['sender_id'].isin(users['user_id'])==False)]
kitties_gen_sold.sort_values(by = 'ymd',ascending = 0, inplace = True)
axiom_direct_sales_cats = kitties_gen_sold['ymd'].value_counts()
temo = kitties_gen_sold.pivot(index = 'join_date', columns = 'generation')


cac_df_2 = events[(events['event'] == 'createsaleauction') & (events['sender_id'].isin(users.user_id))][rel_list_data]