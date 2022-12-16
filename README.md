# Yelp for Ann Arbor

## Environments

```bash
pip install -r requirements.txt
```

If `fastrank` cannot be installed, run

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
then restart your terminal and rerun
```bash
pip install -r requirements.txt
```

## Launch the website

Make sure the packages in `requirements.txt` have been successfully installed. Run

```bash
streamlit run yelp.py
```
The website can be viewed in your browser.

## Do the search

On the webpage, there is a search bar, a slider, a multiselect box, and a checkbox.

You can enter any text you would like to search in the search bar, like restaurant categories, restaurant characteristics, dish names, or restaunrant names. Here are some examples: `fried chicken`, `Subway`, `Thanksgiving dinner`.

You can use the slider to select an integer value from 0 to 5 to filter out restaurants with ratings not less than that value. For example, if you select value 4, then the returned restaurants will only have ratings 4, 4.5, or 5. The slider is initialized to 0. That means if you do not make selection on the slider, the returned results will have any rating from 0 to 5.

In the multiselect box, you can add some features that you would like the restaurants to have. The features include `Takes Reservations`, `Offers Delivery`, `Offers Takeout`, `Good For Kids`, `Waiter Service`, `Accepts Credit Cards`, `Accepts Apple Pay`.

The checkbox is used to filter out the restaurants that are open now.

After finishing the selections, click `Submit` and the results will be returned shortly.
