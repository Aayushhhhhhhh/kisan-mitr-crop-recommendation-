# 🌾 KisanMitr: Smart Crop Recommendation System

A web application that provides intelligent crop recommendations for Indian farmers based on soil health parameters. Built with Streamlit and Machine Learning, KisanMitr helps farmers make data-driven decisions aligned with the National Soil Health Card Schema.

## 🔗 Live Application

**Try it now:** [https://y58evej2wkkkxk8e3jmdks.streamlit.app/](https://y58evej2wkkkxk8e3jmdks.streamlit.app/)

Experience KisanMitr live without any installation required!

## 🎯 Features

- **Interactive Soil Analysis Dashboard**: User-friendly interface to input soil health parameters
- **Machine Learning Recommendations**: Random Forest Classifier with 90%+ accuracy
- **Real-time Predictions**: Instant crop suggestions based on 7 key soil metrics
- **Visual Analytics**: Confusion matrix and performance metrics visualization
- **Data Transparency**: View and download the complete training dataset
- **Contextual Information**: Crop-specific insights and farming recommendations
- **Mobile-Responsive Design**: Works seamlessly across all devices

## 📊 How It Works

The application uses a Random Forest machine learning model trained on comprehensive agricultural data to predict the most suitable crop for given soil conditions. The model considers:

1. **Nitrogen (N)** - 0-140 range
2. **Phosphorus (P)** - 5-145 range
3. **Potassium (K)** - 5-205 range
4. **Temperature** - 8-45°C
5. **Humidity** - 14-100%
6. **pH Level** - 3.5-10.0
7. **Rainfall** - 20-300mm

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Aayushhhhhhhh/kisan-mitr-crop-recommendation-.git
cd kisan-mitr-crop-recommendation-
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to:
```
http://localhost:8501
```

## 📦 Dependencies

```
streamlit
pandas
numpy
seaborn
matplotlib
scikit-learn
```

## 💻 Usage

1. **Launch the Application**: Run `streamlit run app.py`
2. **Input Soil Parameters**: Use the sidebar sliders to enter your soil health card data
3. **Get Recommendations**: Click the "Predict Best Crop 🚜" button
4. **View Results**: See the recommended crop along with contextual farming advice
5. **Explore Analytics**: Check the confusion matrix and model performance metrics

### Example Input

```
Nitrogen (N): 90
Phosphorus (P): 42
Potassium (K): 43
Temperature: 20°C
Humidity: 82%
pH Level: 6.5
Rainfall: 200mm
```

## 🎓 Model Performance

- **Algorithm**: Random Forest Classifier
- **Training Split**: 80% training, 20% testing
- **Accuracy**: ~90%+ on test data
- **Number of Estimators**: 20
- **Supported Crops**: 22 different crop types including rice, wheat, cotton, pulses, fruits, and vegetables

## 📁 Project Structure

```
kisan-mitr-crop-recommendation-/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🌱 Supported Crops

The model can recommend from the following crops:
- **Cereals**: Rice, Wheat, Maize
- **Pulses**: Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil
- **Cash Crops**: Cotton, Jute, Coffee, Coconut
- **Fruits**: Apple, Banana, Grapes, Orange, Papaya, Mango, Muskmelon, Watermelon, Pomegranate

## 🔬 Technical Details

### Machine Learning Pipeline
1. Data loading from curated agricultural dataset
2. Feature engineering and preprocessing
3. Train-test split (80-20 ratio)
4. Random Forest model training
5. Performance evaluation using confusion matrix and accuracy metrics

### Data Source
The application uses a reliable crop recommendation dataset aligned with Indian agricultural standards, sourced from verified agricultural research repositories.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Enhancements

- [ ] Multi-language support (Hindi, regional languages)
- [ ] Weather API integration for real-time climate data
- [ ] Crop yield prediction
- [ ] Fertilizer recommendation system
- [ ] Market price integration
- [ ] Historical trend analysis
- [ ] Mobile app development

## 👨‍💻 Author

**Aayush**
- GitHub: [@Aayushhhhhhhh](https://github.com/Aayushhhhhhhh)

---

**Made with ❤️ for Indian Farmers**

*Empowering agriculture through technology*
