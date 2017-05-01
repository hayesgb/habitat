import os

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(project_folder, 'data')
models_folder = os.path.join(project_folder, 'models')
reports_folder = os.path.join(project_folder, 'reports')

source2_folder = os.path.abspath(r'C:/Users/hayesgr/Dropbox (Personal)/TCHFH and Analyze This/TCHFH Source')
source_folder = os.path.abspath(r'/Users/Greg/Dropbox/TCHFH and Analyze This/TCHFH Source')
analyze_this_materials = os.path.join(r'/Users/Greg/Dropbox/TCHFH and Analyze This', 'Analyze This! Materials')
#contributions = os.path.join(source_folder, 'Analyzer Contributions')
jake_mason = os.path.abspath(r'/Users/Greg/Dropbox/TCHFH and Analyze This/Analyzer Contributions/jake_mason/TCHFH_joined.csv')
ads_folder = os.path.abspath(os.path.join(source_folder, 'ads'))

# Raw source datafiles
online_datafile = os.path.join(source_folder,r'TCHFH_Online_Data_01.23.17.xlsx' )
raisers_edge_datafile = os.path.join(source_folder, r"TCHFH Raiser's Edge Data updated 011917.csv")
habitat_build_datafile = os.path.join(source_folder, r'Habitat Build Addresses.xlsx')
data_dict = os.path.join(source_folder, r'data dictionary.xlsx')
ads_datafile = os.path.join(ads_folder, 'TCHFH_ADS.csv')

# Kevin's Baseline Materials
kevins_baseline_dfile = os.path.join(analyze_this_materials, 'Kevins Baseline Analytic Dataset 2-8-17.csv')


# Interim datafiles
interim_data = os.path.join(data_folder, 'interim')
habitat_dfile = os.path.join(interim_data, 'habitat_build_dfile.csv')
online_dfile = os.path.join(interim_data, 'online_dfile.csv')
raisers_edge_dfile = os.path.join(interim_data, 'raisers_edge_dfile.csv')
joined_dfile = os.path.join(interim_data, 'joined_data.csv')
baseline = os.path.join(interim_data, 'baseline.csv')                               # csv of
zipcode_demographic_data = os.path.join(interim_data, 'zipcode_demographics.csv')  # CSV of demographic data by zipcode
zipcode_processed_demographics = os.path.join(interim_data, 'zipcode_processed_demographics.csv')
redonation_rate_by_gender = os.path.join(interim_data, 'donation_rate_by_gender.csv')
redonation_rate_by_zipcode = os.path.join(interim_data, 'donation_rate_by_zipcode.csv')
redonation_rate_by_marital = os.path.join(interim_data, 'donation_rate_by_marital.csv')
second_donations_by_marital = os.path.join(interim_data, 'second_donations_by_marital.csv') # Second donation amt by marital
second_donations_by_gender = os.path.join(interim_data, 'second_donations_by_gender.csv')   # Second donation amt by gender


# Split datafiles
location_dfile = os.path.join(interim_data, 'location_data.csv')
giving_dfile = os.path.join(interim_data, 'giving_data.csv')
char_dfile = os.path.join(interim_data, 'char_data.csv')

# Cleaned datafiles
cleaned_dfiles = os.path.join(data_folder, 'cleaned_giving_data.csv')
giving = os.path.join(data_folder, 'giving.csv')
personal = os.path.join(data_folder, 'personal.csv')
zipcode_latlng_dictionary = os.path.join(data_folder, 'zipcode_dict.csv')       # Dictionary of zipcodes and corresponding lat/longs
pairwise_distance_matrix = os.path.join(data_folder, 'distances.csv')

# Processed data
processed_data = os.path.join(data_folder, 'processed')
zipcode_dfile = os.path.join(processed_data, 'zipcodes.csv')
pairwise_distance_matrix = os.path.join(processed_data, 'pairwise_distance_matrix.csv') # Pairwise distance matrix
combined_df = os.path.join(processed_data, 'combined_df.csv')
validation_df = os.path.join(processed_data, 'validation_df.csv')
predicted_classes = os.path.join(processed_data, 'predicted_classes.csv')
ts_predicted_classes = os.path.join(processed_data, 'ts_predicted_classes.csv')
predicted_class_probabilities = os.path.join(processed_data, 'predicted_class_probabilities.csv')
complete_training_set = os.path.join(processed_data, 'complete_training_dset.csv')
regression_total = os.path.join(processed_data, 'regression_total.csv')
regression_training = os.path.join(processed_data, 'regression_training.csv')
regression_validation = os.path.join(processed_data, 'regression_validation.csv')

# Training dframes
clf_training_df = os.path.join(processed_data, 'clf_training.csv')

# Trained models
tld_trimmer = os.path.join(models_folder, 'tld_trimmer.pkl')
secondary_domain_trimmer = os.path.join(models_folder, 'secondary_domain_trimmer.pkl')
zipcode_trimmer = os.path.join(models_folder, 'zipcode_trimmer.pkl')
classifier_pipeline = os.path.join(models_folder, 'classifier_pipeline.pkl')  # Classifier pipeline
trained_classifier = os.path.join(models_folder, 'trained_classifier.pkl')    # Trained classifier
polynomial_model = os.path.join(models_folder, 'polynomial_model.pkl')          # Polynomial transformer
outlier_model = os.path.join(models_folder, 'outlier_detector.pkl')             # Trained outlier detector model

nb_classifier_pipeline = os.path.join(models_folder, 'nb_classifier_pipeline.pkl')  # L1_TSS Interactions Classifier pipeline
nb_trained_classifier = os.path.join(models_folder, 'nb_trained_classifier.pkl')    # L1_TSS Trained classifier
#nb_classifier_pipeline = os.path.join(models_folder, 'nb_classifier_pipeline.pkl')  # Classifier pipeline

l1_classifier_pipeline = os.path.join(models_folder, 'l1_classifier_pipeline.pkl')  # L2TSS Classifier pipeline
l1_trained_classifier = os.path.join(models_folder, 'l1_trained_classifier.pkl')    # L2 TSS Trained classifier
#l2_tss_classifier_pipeline = os.path.join(models_folder, 'l2_tss_classifier_pipeline.pkl')  # Classifier pipeline

regression_model = os.path.join(models_folder, 'trained_regressor.pkl')
regression_pipeline = os.path.join(models_folder, 'regression_pipeline.pkl')
regression_polynomial = os.path.join(models_folder, 'regression_polynomial.pkl')
predicted_donation_amts = os.path.join(processed_data, 'predicted_donations.csv')


