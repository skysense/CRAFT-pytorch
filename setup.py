from distutils.core import setup

from setuptools import find_packages

from craft import VERSION


def get_requirements():
    # https://stackoverflow.com/questions/32688688/how-to-write-setup-py-to-include-a-git-repo-as-a-dependency
    with open('requirements.txt') as r:
        requirements = r.read().splitlines()
    required = []

    # do not add to required lines pointing to git repositories
    egg_mark = '#egg='
    for line in requirements:
        if line.startswith('-e git:') or line.startswith('-e git+') or \
                line.startswith('git:') or line.startswith('git+'):
            if egg_mark in line:
                package_name = line[line.find(egg_mark) + len(egg_mark):]
                required.append(package_name + ' @ ' + line)
                # imagededup @ git+ssh://git@github.com/philipperemy/imagededup.git@master#egg=imagededup
            else:
                print('Dependency to a git repository should have the format:')
                print('git+ssh://git@github.com/xxxxx/xxxxxx#egg=package_name')
                exit(1)
        else:
            required.append(line)
    return required


setup(
    name='craft_pytorch',
    version=VERSION,
    packages=find_packages(),
    install_requires=get_requirements(),
    include_package_data=True,
    data_files=[
        ('craft_ic15_20k.pth', ['craft/craft_ic15_20k.pth']),
        ('craft_mlt_25k.pth', ['craft/craft_mlt_25k.pth']),
        ('craft_refiner_CTW1500.pth', ['craft/craft_refiner_CTW1500.pth'])
    ]
)
