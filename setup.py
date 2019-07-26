from distutils.core import setup

setup(
        name='logic_processor',
        version='0.1dev',
        packages=['nn_utils','neural'],
        install_requires=['numpy','matplotlib',
        'tensorflow','keras','pandas','jupyter'],
        author='Ryan Mukai',
        license='MIT'
                            )
