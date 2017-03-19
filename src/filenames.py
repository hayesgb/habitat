import os



project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(project_folder, 'data')
models_folder = os.path.join(project_folder, 'models')


source_folder = os.path.abspath(r'/Users/Greg/Dropbox/TCHFH and Analyze This/TCHFH Source')
analyze_this_materials = os.path.join(r'/Users/Greg/Dropbox/TCHFH and Analyze This', 'Analyze This! Materials')

# Raw source datafiles
online_datafile = os.path.join(source_folder,r'TCHFH_Online_Data_01.23.17.xlsx' )
raisers_edge_datafile = os.path.join(source_folder, r"TCHFH Raiser's Edge Data updated 011917.csv")
habitat_build_datafile = os.path.join(source_folder, r'Habitat Build Addresses.xlsx')
data_dict = os.path.join(source_folder, r'data dictionary.xlsx')

# Kevin's Baseline Materials
kevins_baseline_dfile = os.path.join(analyze_this_materials, 'Kevins Baseline Analytic Dataset 2-8-17.csv')


# Interim datafiles
interim_data = os.path.join(data_folder, 'interim')
habitat_dfile = os.path.join(interim_data, 'habitat_build_dfile.csv')
online_dfile = os.path.join(interim_data, 'online_dfile.csv')
raisers_edge_dfile = os.path.join(interim_data, 'raisers_edge_dfile.csv')
joined_dfile = os.path.join(interim_data, 'joined_data.csv')

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

# Training dframes
clf_training_df = os.path.join(processed_data, 'clf_training.csv')

# Trained models
tld_trimmer = os.path.join(models_folder, 'tld_trimmer.pkl')
secondary_domain_trimmer = os.path.join(models_folder, 'secondary_domain_trimmer.pkl')
zipcode_trimmer = os.path.join(models_folder, 'zipcode_trimmer.pkl')
classifier_pipeline = os.path.join(models_folder, 'classifier_pipeline.pkl')  # Classifier pipeline
trained_classifier = os.path.join(models_folder, 'trained_classifier.pkl')    # Trained classifier


