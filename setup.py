from setuptools import setup
from sphinx.setup_command import BuildDoc
cmdclass={'build_sphinx': BuildDoc}

name = 'mm8'
version = '0.0.1'
release = '0.0.1.0'

setup(
    name=name,
    author='Torben Kimhofer',
    version=release,
    cmdclass=cmdclass,
    packages=['mm8'],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
	    'version': ('setyp.py', version),
	    'release': ('setup.py', release),
	    'source_dir': ('setup.py', 'docs/source') }},
    install_requires=[
        'requests',
        'importlib; python_version == "2.6"',
    ],
)
