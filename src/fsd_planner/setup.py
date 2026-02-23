from setuptools import setup
import os
from glob import glob

package_name = 'fsd_planner'

setup(
    name=package_name,
    version='0.0.0',
    # We must list every sub-folder that contains an __init__.py file 
    # so that 'colcon build' installs them correctly into your workspace.
    packages=[
        package_name,
        package_name + '.lib',
        package_name + '.lib.ft_fsd_path_planning',
    ],
    data_files=[
        # The 'marker' file tells ROS 2 that this is a valid ament_python package
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This line installs your launch files so 'ros2 launch' can find them
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mayuresh',
    maintainer_email='mayuresh@todo.todo',
    description='Path planning wrapper for EUFS using ft-fsd algorithm',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This links the command 'planner_node' to the main function in your script
            'planner_node = fsd_planner.planner_node:main',
        ],
    },
)
