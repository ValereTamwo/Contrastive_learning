from setuptools import setup, find_packages

def _parse_requirements(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Lire les dépendances depuis requirements.txt
install_requires = _parse_requirements('requirements.txt')

setup(
    name='contrastive_learning', 
    version='0.1',  
    description='A contrastive learning project for vision-text alignment',
    author='Franck Tamwo', 
    author_email='franck.tamwo@facsciences-uy1.cm',  
    packages=find_packages(),  # Détecte automatiquement les dossiers contenant __init__.py
    install_requires=install_requires,  # Dépendances depuis requirements.txt
    include_package_data=True,  # Inclut les fichiers de données (ex. dans data/)
    python_requires='>=3.7',  # Version minimale de Python
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
    ],
)