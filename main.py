import click

from src.data.make_dataset2 import main as make_dset
from src.models.regression_model3 import main as regression_model
from src.models.train_model import main as use_classifier
import src.filenames as filenames
import os

@click.group()
def cli():
    '''
    Pick a pipeline.
    '''
    pass

@cli.command()
def make_dataset():
    print('Creating the dataset...')
    make_dset()

@cli.command()
def train_classifier():
    print('Training the classifier...')
    use_classifier(df=os.path.join(filenames.interim_data, 'cross_sectional.csv'), zipcodes=filenames.zipcode_dfile, predict=False)


@cli.command()
def predict_classes():
    print('Making predictions...')
#    use_classifier(predict=True, df=filenames.complete_training_set)
    use_classifier(predict=True, df=filenames.validation_df)

@cli.command()
def train_regression_model():
    print('Training regression model...')
    regression_model(df=filenames.ads_datafile, zipcodes=filenames.zipcode_dfile, predict=False)

@cli.command()
def predict_donation_amt():
    print('Making 2nd donation predictions...')
    regression_model(df=filenames.validation_df, zipcodes=filenames.zipcode_dfile, predict=True)




if __name__ == '__main__':
    cli()