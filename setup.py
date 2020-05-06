from setuptools import setup

setup(name='prepack',
      version='0.3.1',
      description='Python data preparation library',
      url='http://github.com/legale/prepack',
      author='rumi',
      author_email='legale.legale@gmail.com',
      license='MIT',
      packages=['prepack'],
      zip_safe=False,
      install_requires=['numpy','pandas','python-levenshtein'])