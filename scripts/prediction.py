import pandas as pd
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    logging.info("Starting the prediction process.")
    pipeline = joblib.load("output/pipeline.joblib")

    print("Enter the input values for the following features:")
    age_group = int(
        input(
            "What age group do you belong to? (1: 18-22, 2: 23-29, 3: 30-39, 4: 40-49, 5: 50-59, 6: 60-69, 7: 70-79, 8: 80 and over): "
        )
    )
    v144 = int(
        input(
            "In terms of religion, how do you define yourself? (1: Very religious, Haredi, 2: Religious, 3: Traditional religious, 4: Traditional, not so religious, 5: Non-religious, secular): "
        )
    )
    v712 = int(
        input(
            "In our society, where would you rank yourself on this scale nowadays? From Bottom (0) to Top (10): "
        )
    )
    sector = int(input("Which sector do you belong to? (1: Jewish, 2: Arab): "))

    if sector == 2:
        recode_v131 = 0  # Arab
    else:
        v131 = int(
            input(
                "How would you define yourself? (1: Ashkenazi, 2: Sephardic, 3: Mizrachi, 5: Both, 10: Israeli, 99: Other): "
            )
        )
        recode_v131 = {1: 1, 2: 2, 3: 3, 5: 5, 10: 10}.get(
            v131, 99
        )  # Default to 'Other' if not matched

    data = {
        "age_group": [age_group],
        "v144": [v144],
        "v712": [v712],
        "recode_v131": [recode_v131],
        "sector": [sector],
    }

    input_df = pd.DataFrame(data)
    probabilities = pipeline.predict_proba(input_df)
    logging.info(f"Probabilities of voting for each party: {probabilities}")
    parties = [
        "Likud (Benjamin Netanyahu)",
        "Yesh Atid (Yair Lapid)",
        "National Unity (HaMahane HaMamlakhti) (Benny Gantz)",
        "Hatzionut Hadatit (Bezalel Smotrich & Itamar Ben-Gvir)",
        "Shas (Aryeh Deri)",
        "Yahadut HaTorah (Agudat Israel Degel HaTorah) (Moshe Gafni)",
        "Yisrael Beitenu (Avigdor Lieberman)",
        "HaAvoda (Merav Michaeli)",
        "Meretz (Zehava Gal-On)",
        "HaBayit HaYehudi (Ayelet Shaked)",
        "Hadash-Ta’al (Ayman Odeh)",
        "Ra’am (Mansour Abbas)",
        "Balad (Sami Abu Shehadeh)",
    ]

    print("Probabilities of voting for each party:")
    for idx, prob in enumerate(probabilities[0]):
        print(f"{parties[idx]}: {prob:.2f}")


if __name__ == "__main__":
    main()
