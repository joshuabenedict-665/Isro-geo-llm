import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------- Loaders --------------------

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db

def load_district_data():
    with open("data/district_data.json", "r") as f:
        return json.load(f)

# -------------------- District Info --------------------

def search_district_info(query, district_data):
    for district in district_data:
        district_name = district["district"].lower()
        if district_name in query.lower():
            dominant_lulc = district.get("dominant_lulc", "None")
            elevation = district.get("average_elevation", "Unknown")
            return f"""🧠 Answer from District Data:
📍 District: {district["district"]}
🌄 Avg Elevation: {elevation} meters
🗺️ Dominant LULC: {dominant_lulc}"""
    return None

# -------------------- Vector Search --------------------

def retrieve_explanation(query, k=3):
    db = load_vectorstore()
    return db.similarity_search(query, k)

# -------------------- Suitability Rules --------------------

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

    print(f"{district['district']:<20} | Elev: {elevation:>6.1f}m | Built-up: {built_up:>5.1f}% | Barren: {barren_or_waste:>5.1f}% | Forest/Water: {forest_or_water:>5.1f}%")

    # Relaxed and practical condition for urban expansion
    if elevation <= 600 and built_up <= 10 and barren_or_waste >= 5 and forest_or_water <= 50:
        return True
    return False




# -------------------- Main CLI --------------------

if __name__ == "__main__":
    user_query = input("🔍 Enter your question: ").lower()
    district_data = load_district_data()
    
    # Handle suitability questions
    if "suitable for agriculture" in user_query:
        suitable = [d["district"] for d in district_data if is_suitable_for_agriculture(d)]
        label = "Agriculture"

    elif "suitable for solar" in user_query:
        suitable = [d["district"] for d in district_data if is_suitable_for_solar(d)]
        label = "Solar"

    elif "suitable for urban" in user_query or "urban development" in user_query:
        suitable = [d["district"] for d in district_data if is_suitable_for_urban(d)]
        label = "Urban Development"

    else:
        suitable = None
        label = None

    if suitable is not None:
        if suitable:
            print(f"\n🏞️ Suitable Districts for {label}:")
            for name in suitable:
                print("✅", name)
        else:
            print(f"\n⚠️ No suitable districts found for {label}.")
    else:
        # Either district info or tool Q&A
        district_answer = search_district_info(user_query, district_data)
        if district_answer:
            print("\n" + district_answer)
        else:
            print("\n🧠 Answer from Tool Knowledge Base:")
            results = retrieve_explanation(user_query)
            for i, doc in enumerate(results):
                print(f"\nResult {i+1}:\n{doc.page_content}")


