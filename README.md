# Adaptable Individualized Investment Recommendation System


## Overview

This repository contains the code and documentation for my Bachelor Thesis project, "Personalized Stock Recommender." The project was completed in May 2021 and aimed to design a system that provides personalized stock portfolio recommendations based on user interests and risk tolerance.

## Project Details

- **Title:** Adaptable Individualized Investment Recommendation System
- **Completion Date:** May 2021
- **Degree:** Bachelor Thesis

## Features

- Utilized clustering techniques to assess user risk tolerance and preferences.
- Developed a personalized stock portfolio recommendation system based on user interests.
- Deployed the entire system as an end-to-end CI/CD pipeline using Jenkins across Amazon EC2 instance.

## How It Works

The Personalized Stock Recommender uses clustering algorithms to analyze user-provided data, such as historical investment behavior, financial goals, and risk preferences. Based on this information, the system assesses the user's risk tolerance level and identifies their interests in different types of stocks.

Using the collected data, the recommender system generates a personalized stock portfolio that aligns with the user's risk profile and investment preferences.

## Deployment

The entire system has been deployed as an end-to-end CI/CD pipeline, allowing for seamless updates and maintenance. The deployment utilizes Jenkins for continuous integration and continuous deployment (CI/CD) across an Amazon EC2 instance.

## Project Structure

The repository is structured as follows:

1. /.ebextensions -> The '/.ebextensions/00-packages.config' contains configuration files for yum and git.

2. /static -> The /static directory contains all the assets, additional components like images and fonts and data for the project which supports '/templates'.

3. /templates -> The '/templates' directory contains HTML pages or frontend related to the project, such as the sign-in, signup, dashboard, user info etc.

4. /tests 

5. README.md 

6. application.py -> The 'application.py' contains scripts to get the data using twitter API, Cleaning the data, sentiment analysis, recommendation and setup files with variables like API keys or database connections.

7. forest.txt 

8. requirement.txt -> The 'requirement.txt' includes all the necessary packages used in this project.


## Getting Started

To get started with the Personalized Stock Recommender, follow the instructions provided in the `/docs/user_guide` directory. It will guide you through the process of setting up the system, providing user data, and receiving personalized stock portfolio recommendations.

## Feedback and Contributions

Feedback and contributions are welcome! If you have any suggestions, improvements, or bug reports, please open an issue or submit a pull request.

---

Thank you for your interest in the Personalized Stock Recommender project. If you have any questions or need further information, feel free to contact me.

*Author: Vishaal Saravanan, Roobesh Balaji & Mohamed Arshad*
*Contact: vishaals0507@gmail.com*
