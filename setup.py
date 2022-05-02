
from os import path
from setuptools import setup
from setuptools import find_packages
import urllib.request


description = "QAFactEval Summarization Factual Consistency Metric"
f = urllib.request.urlopen("https://raw.githubusercontent.com/salesforce/QAFactEval/master/README.md")
long_description = f.read().decode("utf-8")

setup(name='qafacteval',
      version='0.10',
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/salesforce/QAFactEval',
      author='Alexander R. Fabbri',
      author_email='afabbri@salesforce.com',
      license='BSD',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'qaeval',
      ],
)
