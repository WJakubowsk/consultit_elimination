import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from k_means_constrained import KMeansConstrained
import time


# function that loads data
@st.cache_data
def load_data():
    data = pd.read_excel("Baza_uczniów_CIT_2024.xlsx")
    data.drop(["Imię", "L.P."], axis=1, inplace=True)
    return data


# data preprocessing function
def preprocess_data(data):

    ### operations for categorical columns with order or binary values
    ord_pipeline = Pipeline(
        steps=[
            (
                "encode",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    ### operations for categorical unordered columns
    cat_pipeline = Pipeline(steps=[("encode", OneHotEncoder(handle_unknown="ignore"))])

    ### operations for numerical columns
    num_pipeline = Pipeline(
        steps=[
            (
                "discretize",
                KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform"),
            )
        ]
    )

    # Column transformer
    col_trans = ColumnTransformer(
        transformers=[
            ("ord_pipeline", ord_pipeline, ["Płeć", "Typ prowadzenia zajęć", "Poziom"]),
            ("cat_pipeline", cat_pipeline, ["Język", "Hobby"]),
            ("num_pipeline", num_pipeline, ["Wiek"]),
        ],
        remainder="drop",
        n_jobs=-1,
    )

    # hierarchical clustering at the end of the pipeline (limit as desired number of people in the group)
    model_pipeline = Pipeline([("preprocessing", col_trans)])

    # preprocess data
    data_preprocessed = model_pipeline.fit_transform(data)

    # convert compressed data to numpy array
    decompressed_data = data_preprocessed.toarray()

    return decompressed_data


# function that performs unsupervised clustering
def assign_groups(decompressed_data):

    # run the model that maximises the silhouette score (n_clusters=43)
    kmc = KMeansConstrained(n_clusters=43, size_min=5, size_max=10, random_state=0)
    kmc.fit_predict(decompressed_data)

    return kmc.labels_


# main function used to run the streamlit app demo
def main():
    st.title("Witaj w szkole językowej LinguaViva!")
    st.write("### Przydział grupy wyświetli się poniżej :point_down:")

    # load data
    data = load_data()

    # user input here
    st.sidebar.title("Wypełnij swoje dane:")

    user_data = {}

    user_data["Płeć"] = st.sidebar.selectbox("Wybierz płeć", ["Kobieta", "Mężczyzna"])
    user_data["Wiek"] = st.sidebar.text_input("Wpisz swój wiek", "")
    user_data["Typ prowadzenia zajęć"] = st.sidebar.selectbox(
        "Wybierz typ zajęć", ["Zdalnie", "Stacjonarne"]
    )
    user_data["Poziom"] = st.sidebar.selectbox(
        "Wybierz poziom języka", ["A1", "A2", "B1", "B2", "C1", "C2"]
    )
    user_data["Język"] = st.sidebar.selectbox(
        "Wybierz język",
        ["angielski", "hiszpański", "francuski", "niemiecki", "rosyjski", "włoski"],
    )
    user_data["Hobby"] = st.sidebar.text_input("Wpisz swoje hobby", "")

    if st.button("Zobacz przydział do grupy") and all(
        user_data[col] != "" for col in user_data
    ):
        user_df = pd.DataFrame([user_data])

        # add the user to the dataset
        df = pd.concat([data, user_df], ignore_index=True)

        # preprocess data
        encoded_data = preprocess_data(df)

        # assign groups
        clustered_data = assign_groups(encoded_data)

        # display the group assignments
        # st.write('## Przypisane numery grup')
        # st.write(clustered_data)

        # print the individual group assignment
        st.write(
            "## Zostałeś/aś przydzielony/a do grupy nr ",
            clustered_data[-1],
            " :smiley:",
        )
        st.write(
            "### Będziesz się uczyć z ",
            clustered_data.tolist().count(clustered_data[-1]) - 1,
            " innymi uczniami!",
        )


# run the app
if __name__ == "__main__":
    main()
