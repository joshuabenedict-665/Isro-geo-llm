import streamlit as st
import json
import geopandas as gpd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import leafmap.foliumap as leafmap

# -------------------- Loaders --------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

@st.cache_data
def load_district_data():
    with open("data/district_data.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_geojson():
    return gpd.read_file("data/tamilnadu_districts.geojson")

# -------------------- Suitability Functions --------------------
def is_suitable_for_agriculture(d):
    if d.get("average_elevation", 10000) >= 500:
        return False
    for cls in d.get("lulc_classes", []):
        if "agricultur" in cls["class_name"].lower() and cls["percentage"] > 20:
            return True
    return False

def is_suitable_for_solar(d):
    if d.get("average_elevation", 10000) > 800:
        return False
    for cls in d.get("lulc_classes", []):
        if "wasteland" in cls["class_name"].lower() and cls["percentage"] > 10:
            return True
    return False

def is_suitable_for_urban(district):
    elevation = district.get("average_elevation", 10000)
    built_up = 0
    barren_or_waste = 0
    forest_or_water = 0

    for cls in district.get("lulc_classes", []):
        name = cls["class_name"].lower()
        pct = cls["percentage"]
        if "built" in name:
            built_up += pct
        elif "barren" in name or "wasteland" in name:
            barren_or_waste += pct
        elif "forest" in name or "water" in name:
            forest_or_water += pct

    return elevation <= 600 and built_up <= 10 and barren_or_waste >= 5 and forest_or_water <= 50

# -------------------- Map Display --------------------
def show_map(suitable_districts):
    gdf = load_geojson()
    gdf["District"] = gdf["dtname"].str.lower()  # ✅ fix: use 'dtname'
    filtered = gdf[gdf["District"].isin([d.lower() for d in suitable_districts])]

    m = leafmap.Map(center=[10.8, 78.7], zoom=7)
    m.add_gdf(gdf, layer_name="All Districts", style={"fillOpacity": 0.1, "color": "gray"})
    m.add_gdf(filtered, layer_name="Suitable Districts", style={"fillColor": "#32CD32", "color": "black"}, info_mode="on_hover")
    m.to_streamlit(height=500)


# -------------------- RAG Fallback --------------------
def retrieve_explanation(query, k=3):
    db = load_vectorstore()
    return db.similarity_search(query, k)

# -------------------- UI --------------------
st.set_page_config(page_title="Geo-AI Site Suitability Assistant", layout="centered")
st.title("🛰️ Geo-AI Site Suitability Assistant (Tamil Nadu)")
query = st.text_input("🔍 Ask a geospatial question")

district_data = load_district_data()

if query:
    query_lower = query.lower()

    if "agriculture" in query_lower and "suitable" in query_lower:
        st.subheader("🌾 Suitable Districts for Agriculture")
        result = [d["district"] for d in district_data if is_suitable_for_agriculture(d)]
        if result:
            st.success(", ".join(result))
            show_map(result)
        else:
            st.warning("⚠️ No suitable districts found.")

    elif "solar" in query_lower and "suitable" in query_lower:
        st.subheader("🔆 Suitable Districts for Solar")
        result = [d["district"] for d in district_data if is_suitable_for_solar(d)]
        if result:
            st.success(", ".join(result))
            show_map(result)
        else:
            st.warning("⚠️ No suitable districts found.")

    elif "urban" in query_lower and "suitable" in query_lower:
        st.subheader("🏙️ Suitable Districts for Urban")
        result = [d["district"] for d in district_data if is_suitable_for_urban(d)]
        if result:
            st.success(", ".join(result))
            show_map(result)
        else:
            st.warning("⚠️ No suitable districts found.")

    else:
        st.subheader("📚 Tool Knowledge Base Response")
        results = retrieve_explanation(query)
        for i, doc in enumerate(results):
            st.markdown(f"**Result {i+1}:**\n\n```\n{doc.page_content}\n```")
