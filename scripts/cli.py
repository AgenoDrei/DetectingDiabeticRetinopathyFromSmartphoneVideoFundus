import os.path
import click
import sys
import pandas as pd
import json


@click.group()
@click.version_option()
def cli():
    "Detecting Diabetic Retinopathy from Smartphone Fundus Videos"


@cli.command(name="setup")
@click.argument("dummy")
def setup(dummy):
    """
    :param dummy: mailgun domain that is used
    """
    print('TODO...', dummy)


if __name__ == '__main__':
    cli(sys.argv[1:])

