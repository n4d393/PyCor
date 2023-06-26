# Welcome to PyCor2 :wave:

**The fastest way to determine your client stay type.**

![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHNOkSAR9mWWW6K64hobZFA0vPGQf_eCljxw&usqp=CAU "Leisure🏖️ or Business💼?")

Welcome to the future of hotel management with our state-of-the-art Streamlit app, PyCor.
Our app allows hotel owners to easily determine the stay type (Leisure vs Business) of their clients with incredible accuracy using a pre-trained machine learning model.
Leverage your internal booking data to determine the stay type of your clients in a few minutes.

With this powerful tool, hotel owners can improve their marketing and customer service strategies, leading to increased customer satisfaction and revenue. It's time to take your hotel to the next level!

## Getting Started
Ready to revolutionize your hotel business? Here's what you need to do to get started:

### Prerequisites
* Python 3
* Streamlit
* pandas
* numpy
* sklearn

### Installing
First, clone the repository to your local machine:

```zsh
git clone https://github.com/Ben-Taarit/PyCor.git hotel-stay-type-classifier
```

Navigate to the project directory:

```zsh
cd hotel-stay-type-classifier
```

Next, install the required packages:

```zsh
pip install -r requirements.txt
```

### Running the app
To run the app, simply execute the following command:

```zsh
streamlit run app/main.py
```

***Voila! The app will be available at http://localhost:8501/***



### Input
The app takes the following input:

* Age
* Gender
* Family
* Customer Type
* Reservation Status
* Country
* Market Segment
* Distribution Channel
...

### Output
The app will output the predicted stay type (Leisure or Business) based on the input provided. It's that simple!

## Built With

[Streamlit](https://streamlit.io/)  - The web framework we used
[Scikit-learn](https://scikit-learn.org/stable/) - Machine Learning library

## Authors
 * [Jean-Baptiste Pinede](jeanbaptiste_pinede@yahoo.fr)
 * [Nadege Achtergaele](n4d393@gmail.com)


## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
* A special thanks to our team for creating such an amazing solution for hotel owners.
* We express our gratitude to [Accor](https://group.accor.com/en) for providing us with valuable data, making this project possible.
* We extend our appreciation to the whole [DataScientest](https://datascientest.com/) team for enabling the successful completion of this project.
* Inspiration from all the hotel owners who were looking for a solution like this.
* And to the future of hotel management!

Don't wait any longer, give PyCor a try and experience the ultimate solution for determining client stay type. Your hotel will thank you!



[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]()
