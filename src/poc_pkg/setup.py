from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'poc_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.*')))
    ],
    include_package_data=True,        # ← важная строка
    package_data={                    # ← говорим, что в пакете есть бинарные файлы
        # ключ — имя модуля/пакета, значение — список шаблонов файлов
        package_name: ['*.so'],       
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'api_server = poc_pkg.api_server:main',
            'arm_node = poc_pkg.arm_node:main',
        ],
    },
)
