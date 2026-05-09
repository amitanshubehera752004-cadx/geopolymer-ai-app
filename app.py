import gdown
import os
import streamlit as st
import numpy as np
import pickle
import joblib
import pandas as pd
import pygad

# ===============================
# DOWNLOAD LARGE FILES FROM GOOGLE DRIVE
# ===============================
def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# 🔥 Your files (from Google Drive)
download_file("1j2nbeNE8bYXayXLDLB4hBS17r2y7GmGr", "forward_models.pkl")
download_file("1m62q4kL_jci8LfzpAafvGKU_GQcFJXPp", "forward_model.pkl")
download_file("1uTAouTup5QwMS8pXziRS3xVudz3HUXQC", "prop_models.pkl")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Geopolymer AI Tool",
    page_icon="🧱",
    layout="wide"
)

# ===============================
# LOAD MODELS
# ===============================
f_model = pickle.load(open("forward_model.pkl", "rb"))
f_scaler = pickle.load(open("forward_scaler.pkl", "rb"))

prop_models = joblib.load("prop_models.pkl")
forward_models = joblib.load("forward_models.pkl")
cost_model = joblib.load("cost_model.pkl")
fixed_values = joblib.load("fixed_values.pkl")

cols = joblib.load("columns.pkl")
opt_cols = cols["opt_cols"]
fixed_cols = cols["fixed_cols"]
properties = cols["properties"]

# ===============================
# HEADER
# ===============================
st.title("🧱 AI-Based Geopolymer Concrete Mix Designer")

st.markdown("""
Design and optimize geopolymer concrete using AI.

✔ **Forward Mode** → Predict properties from mix  
✔ **Reverse Mode** → Generate mix from desired properties  
✔ **Cost estimation included**
""")

mode = st.selectbox(
    "Select Mode",
    ["Forward Prediction", "Reverse Prediction"]
)

# ===============================
# FORWARD PREDICTION
# ===============================
if mode == "Forward Prediction":

    st.header("🔧 Enter Mix Parameters")

    col1, col2 = st.columns(2)

    with col1:
        fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0)
        ggbs = st.number_input("GGBFS (kg/m³)", min_value=0.0)
        naoh_molarity = st.number_input("NaOH Molarity (M)", min_value=0.0)
        naoh_amt = st.number_input("NaOH Amount (kg/m³)", min_value=0.0)
        na2sio3_amt = st.number_input("Na2SiO3 Amount (kg/m³)", min_value=0.0)
        extra_water = st.number_input("Extra Water (kg/m³)", min_value=0.0)

    with col2:
        coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0)
        fine_agg = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0)
        recycled_agg = st.number_input("Recycled Aggregate (kg/m³)", min_value=0.0)
        curing_temp = st.number_input("Curing Temperature (°C)", min_value=0.0)
        curing_time = st.number_input("Curing Time (hours)", min_value=0.0)
        age = st.number_input("Age of Testing (days)", min_value=0.0)

    if st.button("🚀 Predict Properties"):

        with st.spinner("🔍 Predicting properties..."):

            input_data = np.array([[
                fly_ash, ggbs, naoh_molarity, naoh_amt,
                na2sio3_amt, extra_water, coarse_agg,
                fine_agg, recycled_agg, curing_temp,
                curing_time, age
            ]])

            input_scaled = f_scaler.transform(input_data)
            prediction = f_model.predict(input_scaled)
            prediction = np.array(prediction).flatten()

        st.subheader("📊 Predicted Properties")

        c1, c2, c3 = st.columns(3)

        c1.metric("Compressive Strength", f"{prediction[0]:.2f} MPa")
        c2.metric("Rebound Hardness", f"{prediction[1]:.2f}")
        c3.metric("UPV", f"{prediction[2]:.2f} m/s")

        c1.metric("Slump", f"{prediction[3]:.2f} mm")
        c2.metric("Split Tensile", f"{prediction[4]:.2f} MPa")
        c3.metric("Flexural Strength", f"{prediction[5]:.2f} MPa")

# ===============================
# REVERSE PREDICTION
# ===============================
else:

    st.header("🎯 Enter Target Properties")

    col1, col2 = st.columns(2)

    with col1:
        cs = st.number_input("Compressive Strength (MPa)", min_value=0.0)

    with col2:
        slump_input = st.text_input("Slump (mm) [optional]")
        fs_input = st.text_input("Flexural Strength (MPa) [optional]")

    if st.button("⚙️ Generate Mix Design and Cost"):

        with st.spinner("⚡ Optimizing mix using AI (GA)..."):

            slump = float(slump_input) if slump_input else None
            fs = float(fs_input) if fs_input else None

            known = {
                "Compressive Strength (Mpa)": cs,
                "RH (rebound hardness)": None,
                "UPV (m/s)": None,
                "SLUMP (mm)": slump,
                "STS (split tensile strength)": None,
                "FS (flexural strength)": fs
            }

            for prop in properties:
                if known[prop] is None:

                    inputs = []
                    for p in properties:
                        if p != prop:
                            val = known[p] if known[p] is not None else 0
                            inputs.append(val)

                    input_df = pd.DataFrame([inputs], columns=[p for p in properties if p != prop])
                    pred = prop_models[prop].predict(input_df)[0]
                    known[prop] = pred

            target = np.array([known[p] for p in properties])

            bounds = [
                (200,450),(50,300),(4,14),(30,120),
                (60,250),(20,100),(900,1300),(400,800),(0,300)
            ]

            gene_space = [{'low':b[0],'high':b[1]} for b in bounds]

            def fitness_func(ga, solution, idx):
                sol_df = pd.DataFrame([dict(zip(opt_cols, solution))])
                for col in fixed_cols:
                    sol_df[col] = fixed_values[col]

                preds = np.array([forward_models[p].predict(sol_df)[0] for p in properties])
                error = np.mean(np.abs((preds - target) / (target + 1e-6)))
                return -error

            ga = pygad.GA(
                num_generations=30,
                sol_per_pop=20,
                num_parents_mating=8,
                num_genes=len(opt_cols),
                gene_space=gene_space,
                fitness_func=fitness_func,
                mutation_num_genes=2,
                random_seed=42
            )

            ga.run()
            solution, _, _ = ga.best_solution()

            best_mix = dict(zip(opt_cols, solution))
            for col in fixed_cols:
                best_mix[col] = fixed_values[col]

            best_df = pd.DataFrame([best_mix])
            predicted_cost = cost_model.predict(best_df)[0]

        # ================= OUTPUT =================

        st.subheader("🧪 Final Properties")
        cols_prop = st.columns(3)
        for i, prop in enumerate(properties):
            cols_prop[i % 3].metric(prop, f"{known[prop]:.2f}")

        st.subheader("📦 Mix Design")
        mix_df = pd.DataFrame(best_mix.items(), columns=["Parameter", "Value"])
        st.dataframe(mix_df, use_container_width=True)

        st.success(f"💰 Estimated Cost: ₹{predicted_cost:.2f} per m³")
