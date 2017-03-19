import click

from data.make_dataset import main as make_dset
from models.train_model import main as train_classifier
import filenames as filenames

@click.group()
def cli():
    '''
    Pick a pipeline.
    '''
    pass

@cli.command()
def make_dataset():
    print('Creating the cleaned dataset...')
    make_dset(file1=filenames.online_datafile, file2=filenames.raisers_edge_datafile,
              output_filepath=filenames.combined_df)

@cli.command()
def train_clf():
    print('Training the classifier...')
    train_classifier()


@cli.command()
def predict_classes():
    print('Making predictions...')
    train_classifier(predict=True)


if __name__ == '__main__':
    cli()