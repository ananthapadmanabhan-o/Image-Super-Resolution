from setuptools import setup,find_packages

VERSION = '0.0.1'
AUTHOR_NAME = 'ananthapadmanabhan-o'
AUTHOR_EMAIL = 'ananthan51ah@gmail.com'
SRC_REPO = 'srgan'
REPO_NAME = 'Image-Super-Resolution'



setup(
    name=SRC_REPO,
    version=VERSION,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    url=f'https://github.com/{AUTHOR_NAME}/{REPO_NAME}',
    package_dir={'':'.'},
    packages=find_packages(where='.')
)