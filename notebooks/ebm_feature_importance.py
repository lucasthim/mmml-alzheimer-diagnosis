
ebm = None


len(ebm.additive_terms_)
len(ebm.bagged_models_)
ebm.feature_importances_
ebm.feature_names
ebm.feature_groups_
ebm.intercept_
ebm.predict_and_contrib(df_test.drop('DIAGNOSIS',axis=1)[:1])
ebm_local = ebm.explain_local(df_test.drop('DIAGNOSIS',axis=1),df_test['DIAGNOSIS'])
