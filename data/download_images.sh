#!/usr/bin/env bash
sudo apt-get install -y fastjar
wget https://ivc.ischool.utexas.edu/dataset_downloads/VizWiz_answer_difference_images.zip
jar xvf VizWiz_answer_difference_images.zip
rm VizWiz_answer_difference_images.zip
