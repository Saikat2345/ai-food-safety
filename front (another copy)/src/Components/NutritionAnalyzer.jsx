import React, { useState, useRef } from 'react';
import { Upload, Camera, Loader2, CheckCircle, AlertTriangle, XCircle, BarChart3 } from 'lucide-react';

const NutritionAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
      
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(selectedFile);
    } else {
      setError('Please select a valid image file (JPEG, PNG)');
      setFile(null);
      setPreview(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    handleFileSelect(droppedFile);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const analyzeImage = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed. Please try again.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze image. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getClassificationColor = (classification) => {
    switch (classification) {
      case 'Safe':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'Harmful':
        return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'Very Harmful':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getClassificationIcon = (classification) => {
    switch (classification) {
      case 'Safe':
        return <CheckCircle className="w-5 h-5" />;
      case 'Harmful':
        return <AlertTriangle className="w-5 h-5" />;
      case 'Very Harmful':
        return <XCircle className="w-5 h-5" />;
      default:
        return null;
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-3 rounded-2xl">
              <BarChart3 className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            Nutrition Label Analyzer
          </h1>
          <p className="text-gray-600 text-lg">
            Upload a photo of any nutrition label to get instant health insights
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-white rounded-3xl shadow-xl p-8 border border-gray-100">
              <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                <Camera className="w-6 h-6 mr-2 text-blue-500" />
                Upload Image
              </h2>
              
              {!preview ? (
                <div
                  className="border-2 border-dashed border-gray-300 rounded-2xl p-12 text-center hover:border-blue-400 transition-colors cursor-pointer bg-gray-50 hover:bg-blue-50"
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-lg font-medium text-gray-600 mb-2">
                    Drop your image here or click to browse
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports JPG, PNG, and other image formats
                  </p>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={(e) => handleFileSelect(e.target.files[0])}
                    accept="image/*"
                    className="hidden"
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-2xl overflow-hidden bg-gray-100">
                    <img
                      src={preview}
                      alt="Preview"
                      className="w-full h-64 object-contain"
                    />
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={analyzeImage}
                      disabled={loading}
                      className="flex-1 bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-6 rounded-xl font-medium hover:from-blue-600 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin mr-2" />
                          Analyzing...
                        </>
                      ) : (
                        'Analyze Nutrition'
                      )}
                    </button>
                    <button
                      onClick={reset}
                      className="px-6 py-3 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 transition-colors"
                    >
                      Reset
                    </button>
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center text-red-700">
                  <XCircle className="w-5 h-5 mr-2" />
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result ? (
              <div className="bg-white rounded-3xl shadow-xl p-8 border border-gray-100">
                <h2 className="text-2xl font-semibold text-gray-800 mb-6">
                  Analysis Results
                </h2>

                {/* Classification Badge */}
                <div className={`inline-flex items-center px-4 py-2 rounded-full border-2 font-medium mb-6 ${getClassificationColor(result.class)}`}>
                  {getClassificationIcon(result.class)}
                  <span className="ml-2">{result.class}</span>
                  <span className="ml-2 text-sm opacity-75">
                    ({result.score}/100)
                  </span>
                </div>

                {/* Message */}
                <div className="mb-6">
                  <h3 className="font-semibold text-gray-800 mb-2">Assessment:</h3>
                  <p className="text-gray-600">{result.message}</p>
                </div>

                {/* Better Product Suggestion */}
                {result.better_product && (
                  <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-xl">
                    <h3 className="font-semibold text-green-800 mb-2">💡 Suggestion:</h3>
                    <p className="text-green-700">{result.better_product}</p>
                  </div>
                )}

                {/* Nutrition Data */}
                <div className="space-y-4">
                  <h3 className="font-semibold text-gray-800">Nutrition Facts:</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-500">Calories</div>
                      <div className="font-semibold">{result.nutrition_data.calories} kcal</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-500">Sugar</div>
                      <div className="font-semibold">{result.nutrition_data.sugar}g</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-500">Saturated Fat</div>
                      <div className="font-semibold">{result.nutrition_data.sat_fat}g</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-500">Sodium</div>
                      <div className="font-semibold">{result.nutrition_data.sodium}mg</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-500">Fiber</div>
                      <div className="font-semibold">{result.nutrition_data.fiber}g</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-500">Protein</div>
                      <div className="font-semibold">{result.nutrition_data.protein}g</div>
                    </div>
                  </div>
                </div>

                {/* Extracted Text (for debugging) */}
                {result.extracted_text && (
                  <details className="mt-6">
                    <summary className="text-sm font-medium text-gray-500 cursor-pointer">View extracted text</summary>
                    <div className="mt-2 p-3 bg-gray-50 rounded-lg text-sm text-gray-600">
                      {result.extracted_text}
                    </div>
                  </details>
                )}
              </div>
            ) : (
              <div className="bg-white rounded-3xl shadow-xl p-8 border border-gray-100">
                <h2 className="text-2xl font-semibold text-gray-800 mb-6">
                  How it works
                </h2>
                <div className="space-y-4">
                  <div className="flex items-start">
                    <div className="bg-blue-100 text-blue-600 rounded-full p-2 mr-4 mt-1">
                      <span className="text-sm font-bold">1</span>
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-800">Upload Image</h3>
                      <p className="text-gray-600 text-sm">Take a clear photo of the nutrition label</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="bg-blue-100 text-blue-600 rounded-full p-2 mr-4 mt-1">
                      <span className="text-sm font-bold">2</span>
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-800">AI Analysis</h3>
                      <p className="text-gray-600 text-sm">Our AI extracts and analyzes nutrition data</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="bg-blue-100 text-blue-600 rounded-full p-2 mr-4 mt-1">
                      <span className="text-sm font-bold">3</span>
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-800">Get Results</h3>
                      <p className="text-gray-600 text-sm">Receive health classification and recommendations</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NutritionAnalyzer;