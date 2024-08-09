// pages/index.js
"use client"
import React from 'react';
import  { useState } from 'react';
import dynamic from 'next/dynamic';
import 'tailwindcss/tailwind.css';
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });
const data = [
  { id: 1, name: 'Alice', age: 25, city: 'New York' },
  { id: 2, name: 'Bob', age: 30, city: 'San Francisco' },
  { id: 3, name: 'Charlie', age: 35, city: 'Los Angeles' },
];

const Home = () => {
  const [theme, setTheme] = useState('dark');
  const [targetColumn, setTargetColumn] = useState('');
  const [predictionType, setPredictionType] = useState('range');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [predictionResults, setPredictionResults] = useState([]);
  const [predictionChart, setPredictionChart] = useState({});
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [selectedFields, setSelectedFields] = useState([]);
  const [reportData, setReportData] = useState(null);
  const handleTrainModel = () => {
    // Dummy data for training model
    console.log('Training model with target column:', targetColumn);
  };
  const handleFieldChange = (field) => {
    setSelectedFields((prev) =>
      prev.includes(field) ? prev.filter((f) => f !== field) : [...prev, field]
    );
  };
  const handleGenerateReport = () => {
    // Dummy data for report generation
    const data = {
      fields: selectedFields,
      results: [
        { date: '2024-01-01', Field1: 10, Field2: 20 },
        { date: '2024-01-02', Field1: 15, Field2: 25 },
      ],
    };
    setReportData(data);
  };

  const availableFields = ['Field1', 'Field2', 'Field3', 'Field4'];
  const handleGenerateForecast = () => {
    // Dummy data for generating forecast
    setPredictionResults([
      { date: '2024-01-01', prediction: 10 },
      { date: '2024-01-02', prediction: 15 },
    ]);

    setPredictionChart({
      data: [
        {
          x: ['2024-01-01', '2024-01-02'],
          y: [10, 15],
          type: 'scatter',
          mode: 'lines+markers',
          marker: { color: 'blue' },
        },
      ],
      layout: { title: 'Prediction Chart' },
    });
  };


  return (
    <>    <div className={`container mx-auto p-6 ${theme === 'dark' ? 'bg-black text-white' : 'bg-blue-100 text-black'}`}>
      <h1 className="text-3xl font-bold mb-6">Data Transformation and Visualization</h1>
      
      {/* <div className="flex justify-end mb-4">
        <button
          className="py-2 px-4 rounded bg-blue-500 text-white hover:bg-blue-700"
          onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
        >
          Toggle Theme
        </button>
      </div> */}

      <div id="data-table" className="overflow-x-auto">
        <table className="min-w-full border rounded">
          <thead className="bg-gray-800 ">
            <tr>
              <th className="w-1/4 py-2">ID</th>
              <th className="w-1/4 py-2">Name</th>
              <th className="w-1/4 py-2">Age</th>
              <th className="w-1/4 py-2">City</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item) => (
              <tr key={item.id} className="text-center border-b">
                <td className="py-2">{item.id}</td>
                <td className="py-2">{item.name}</td>
                <td className="py-2">{item.age}</td>
                <td className="py-2">{item.city}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div id="insights-container" className="mt-8">
        <h2 className="text-2xl font-bold mb-4">Insights</h2>
        <p>Dummy insights data will be displayed here.</p>
      </div>

      <div id="visualizations-container" className="mt-8">
        <h2 className="text-2xl font-bold mb-4">Visualizations</h2>
        <p>Dummy visualizations data will be displayed here.</p>
      </div>

      <div className=" p-5 min-h-screen">
      <div className="container mx-auto">
        <h2 className="text-2xl font-bold mb-5">Machine Learning Model Results</h2>
        <div className="actions mb-10">
          <div className="form-group mb-5">
            <label htmlFor="target-column" className="block text-xl mb-2">Select Target Column:</label>
            <select
              id="target-column"
              name="target-column"
              className="p-2 border rounded w-full"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
            >
              <option value="">Select Column</option>
              <option value="Column1">Column1</option>
              <option value="Column2">Column2</option>
              <option value="Column3">Column3</option>
            </select>
          </div>
          <button
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-700 mr-2"
            onClick={handleTrainModel}
          >
            Train ML Model
          </button>
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700"
            onClick={handleGenerateForecast}
          >
            Generate Forecast
          </button>
        </div>
        <div id="nn-performance" className="mb-10">
          <h3 className="text-xl mb-3">NN Performance</h3>
          <p>Dummy performance metrics...</p>
        </div>
        <div id="nn-sample-predictions" className="mb-10">
          <h3 className="text-xl mb-3">NN Sample Predictions</h3>
          {predictionResults.length > 0 ? (
            <ul>
              {predictionResults.map((result, index) => (
                <li key={index} className="mb-2">
                  {result.date}: {result.prediction}
                </li>
              ))}
            </ul>
          ) : (
            <p>No predictions available.</p>
          )}
        </div>
        <div id="nn-scatter-plot" className="mb-10">
          <h3 className="text-xl mb-3">NN Scatter Plot</h3>
          {predictionChart.data ? (
            <Plot
              data={predictionChart.data}
              layout={predictionChart.layout}
              className="w-full"
            />
          ) : (
            <p>No chart data available.</p>
          )}
        </div>
        <div id="prediction-form" className="mb-10">
          <h3 className="text-xl mb-3">Make Predictions</h3>
          <div className="form-group mb-5">
            <label htmlFor="prediction-type" className="block mb-2">Prediction Type:</label>
            <select
              id="prediction-type"
              className="p-2 border rounded w-full"
              value={predictionType}
              onChange={(e) => setPredictionType(e.target.value)}
            >
              <option value="range">Date Range Prediction</option>
            </select>
          </div>
          <div className="form-group mb-5">
            <label htmlFor="start-date" className="block mb-2">Start Date:</label>
            <input
              type="date"
              id="start-date"
              className="p-2 border rounded w-full"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div className="form-group mb-5">
            <label htmlFor="end-date" className="block mb-2">End Date:</label>
            <input
              type="date"
              id="end-date"
              className="p-2 border rounded w-full"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
          <button
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-700"
            onClick={handleGenerateForecast}
          >
            Generate Forecast
          </button>
        </div>
        <div id="prediction-container">
          <h2 className="text-2xl mb-5">Prediction Results</h2>
          <div id="prediction-results" className="mb-5">
            {predictionResults.length > 0 ? (
              <ul>
                {predictionResults.map((result, index) => (
                  <li key={index} className="mb-2">
                    {result.date}: {result.prediction}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No prediction results available.</p>
            )}
          </div>
          <div id="prediction-chart" className="w-full">
            {predictionChart.data ? (
              <Plot
                data={predictionChart.data}
                layout={predictionChart.layout}
                className="w-full"
              />
            ) : (
              <p>No chart data available.</p>
            )}
          </div>
        </div>
      </div>
    </div>
    <div className="  p-5 min-h-screen">
      <div className="container mx-auto">
        <h2 className="text-2xl font-bold mb-5">Generate Advanced Report</h2>
        <div className="mb-10">
          <div className="form-group mb-5">
            <label htmlFor="start-date" className="block text-xl mb-2">Start Date:</label>
            <input
              type="date"
              id="start-date"
              className="p-2 border rounded w-full"
              value={dateRange.start}
              onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
            />
          </div>
          <div className="form-group mb-5">
            <label htmlFor="end-date" className="block text-xl mb-2">End Date:</label>
            <input
              type="date"
              id="end-date"
              className="p-2 border rounded w-full"
              value={dateRange.end}
              onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
            />
          </div>
          <div className="form-group mb-5">
            <label className="block text-xl mb-2">Select Fields:</label>
            {availableFields.map((field) => (
              <div key={field} className="mb-2">
                <input
                  type="checkbox"
                  id={field}
                  className="mr-2"
                  checked={selectedFields.includes(field)}
                  onChange={() => handleFieldChange(field)}
                />
                <label htmlFor={field}>{field}</label>
              </div>
            ))}
          </div>
          <button
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-700"
            onClick={handleGenerateReport}
          >
            Generate Report
          </button>
        </div>
        <div id="report-container" className="mb-10">
          <h3 className="text-xl mb-3">Report Results</h3>
          {reportData ? (
            <table className="min-w-full ">
              <thead>
                <tr>
                  <th className="py-2 px-4 border-b">Date</th>
                  {reportData.fields.map((field) => (
                    <th key={field} className="py-2 px-4 border-b">{field}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {reportData.results.map((result, index) => (
                  <tr key={index}>
                    <td className="py-2 px-4 border-b">{result.date}</td>
                    {reportData.fields.map((field) => (
                      <td key={field} className="py-2 px-4 border-b">{result[field]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p>No report data available.</p>
          )}
        </div>
      </div>
    </div>

    </div>
    
    </>
  );
};

export default Home;
