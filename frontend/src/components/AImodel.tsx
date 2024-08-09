// pages/index.js

import { useState, useEffect } from 'react';
import { Tabulator } from 'tabulator-tables';
import Plotly from 'plotly.js-dist';
import io from 'socket.io-client';
import Head from 'next/head';
import styles from '../styles/Home.module.css';

const Home = () => {
  const [columns, setColumns] = useState([]);
  const [categoricalColumns, setCategoricalColumns] = useState([]);
  const [initialData, setInitialData] = useState([]);
  const [totalRows, setTotalRows] = useState(0);
  const [errorMessages, setErrorMessages] = useState('');
  const [successMessages, setSuccessMessages] = useState('');
  const [warningMessages, setWarningMessages] = useState('');
  const [loading, setLoading] = useState(false);
  const [progressMessage, setProgressMessage] = useState('');
  const [advancedResults, setAdvancedResults] = useState([]);
  const [insights, setInsights] = useState('');
  const [visualizations, setVisualizations] = useState([]);
  const [nnPerformance, setNnPerformance] = useState('');
  const [nnSamplePredictions, setNnSamplePredictions] = useState('');
  const [nnScatterPlot, setNnScatterPlot] = useState(null);
  const [predictionResults, setPredictionResults] = useState('');
  const [predictionChart, setPredictionChart] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [reportProgress, setReportProgress] = useState(0);
  const [reportDownloadLink, setReportDownloadLink] = useState('');

  useEffect(() => {
    fetchColumnsData();
    fetchInitialData();
    fetchCategoricalColumnsData();
    fetchTotalRowsData();
  }, []);

  const fetchColumnsData = async () => {
    const res = await fetch('/api/columns');
    const data = await res.json();
    setColumns(data);
  };

  const fetchInitialData = async () => {
    const res = await fetch('/api/initial-data');
    const data = await res.json();
    setInitialData(data);
  };

  const fetchCategoricalColumnsData = async () => {
    const res = await fetch('/api/categorical-columns');
    const data = await res.json();
    setCategoricalColumns(data);
  };

  const fetchTotalRowsData = async () => {
    const res = await fetch('/api/total-rows');
    const data = await res.json();
    setTotalRows(data);
  };

  const applyTransformations = () => {
    // Handle transformations logic
  };

  const applyAdvancedTransformations = () => {
    // Handle advanced transformations logic
  };

  const generateInsights = () => {
    // Handle generating insights
  };

  const generateVisualizations = () => {
    // Handle generating visualizations
  };

  const trainModel = () => {
    // Handle training model
  };

  const makePredictions = () => {
    // Handle making predictions
  };

  const handleSendMessage = () => {
    // Handle sending chat messages
  };

  const generateReport = () => {
    // Handle report generation
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>Data Transformation and Visualization</title>
        <link href="https://unpkg.com/tabulator-tables@5.1.7/dist/css/tabulator.min.css" rel="stylesheet" />
      </Head>

      <h1>Data Transformation and Visualization</h1>

      <div id="error-messages" className={styles.errorContainer}>{errorMessages}</div>
      <div id="success-messages" className={styles.successContainer}>{successMessages}</div>
      <div id="warning-messages" className={styles.warningContainer}>{warningMessages}</div>
      <div id="loading-indicator" className={styles.loading}>{loading && 'Processing...'}</div>
      <div id="progress-message">{progressMessage}</div>

      <div className={styles.transformations}>
        <h2>Apply Transformations</h2>
        <form id="transformation-form">
          <div className="form-group">
            <h3>Select Columns to separate Datetime:</h3>
            {columns.map((column, index) => (
              <div key={index}>
                <input type="checkbox" id={`datetime-${column.replace(' ', '_')}`} name="datetime-columns" value={column} />
                <label htmlFor={`datetime-${column.replace(' ', '_')}`}>{column}</label>
              </div>
            ))}
          </div>

          <div className="form-group">
            <h3>Select Columns to separate Numbers:</h3>
            {columns.map((column, index) => (
              <div key={index}>
                <input type="checkbox" id={`currency-${column.replace(' ', '_')}`} name="currency-columns" value={column} />
                <label htmlFor={`currency-${column.replace(' ', '_')}`}>{column}</label>
              </div>
            ))}
          </div>

          <div className="form-group">
            <h3>Select Columns to separate Them:</h3>
            {columns.map((column, index) => (
              <div key={index}>
                <input type="checkbox" id={`address-${column.replace(' ', '_')}`} name="address-columns" value={column} />
                <label htmlFor={`address-${column.replace(' ', '_')}`}>{column}</label>
              </div>
            ))}
          </div>

          <div className="form-group">
            <h3>Select Columns for Categorical Data:</h3>
            {categoricalColumns.map((column, index) => (
              <div key={index}>
                <input type="checkbox" id={`categorical-${column.replace(' ', '_')}`} name="categorical-columns" value={column} />
                <label htmlFor={`categorical-${column.replace(' ', '_')}`}>{column}</label>
              </div>
            ))}
          </div>

          <div className="form-group">
            <h3>Set Similarity Threshold for Duplicate Detection: <span id="threshold-value">100</span></h3>
            <input type="range" id="similarity-threshold" name="similarity-threshold" min="0" max="100" value="100" />
          </div>

          <button type="button" id="apply-transformations" onClick={applyTransformations}>Apply Transformations</button>
        </form>
      </div>

      <div id="data-table"></div>

      <div id="advanced-transforms">
        <h3>Advanced Transformations:</h3>
        <label><input type="checkbox" name="advanced-transforms" value="high_dimensionality" /> Handle High Dimensionality</label>
        <label><input type="checkbox" name="advanced-transforms" value="scale_transform" /> Scale and Transform Data</label>
        <label><input type="checkbox" name="advanced-transforms" value="feature_engineering" /> Perform Feature Engineering</label>
        <label><input type="checkbox" name="advanced-transforms" value="dynamic_feature" /> Perform Dynamic Feature Transformation</label>
        <label><input type="checkbox" name="advanced-transforms" value="advanced_transform" /> Perform Advanced Data Transformation</label>
        <button id="apply-advanced-transforms" onClick={applyAdvancedTransformations}>Apply Advanced Transformations</button>
      </div>

      <div id="advanced-results">
        <h2>Advanced Transformation Results</h2>
        {advancedResults.map((result, index) => (
          <div key={index}>
            <h3>{result.title}</h3>
            <p>{result.content}</p>
          </div>
        ))}
      </div>

      <div className={styles.actions}>
        <button id="generate-insights" onClick={generateInsights}>Generate Insights</button>
        <button id="generate-visualizations" onClick={generateVisualizations}>Generate Visualizations</button>
        <div className="form-group">
          <label htmlFor="target-column">Select Target Column:</label>
          <select id="target-column" name="target-column">
            {columns.map((column, index) => (
              <option key={index} value={column}>{column}</option>
            ))}
          </select>
        </div>
        <button id="train-model" onClick={trainModel}>Train ML Model</button>
        <button id="make-predictions" disabled>Make Predictions</button>
        <a href="#" id="download-button" className={`${styles.button} ${styles.disabled}`}>Download Cleaned Data</a>
        <button id="save-state">Save State</button>
        <button id="load-state">Load State</button>
        <button id="show-report-generation" onClick={generateReport}>Generate Advanced Report</button>
      </div>

      <div id="insights-container">
        <h2>Insights</h2>
        <div id="insights">{insights}</div>
      </div>

      <div id="visualizations-container">
        <h2>Visualizations</h2>
        {/* {visualizations.map((viz, index) => (
          <div key={index} className={styles.visualizationWrapper}>
            <h3>{viz.title}</h3>
         <div id={viz.id} className={styles}>            <div id={viz.id} className={styles.visualization}></div>
          </div>
        ))} */}
      </div>

      <div id="nn-performance">
        <h2>Neural Network Performance</h2>
        <div id="performance">{nnPerformance}</div>
      </div>

      <div id="nn-sample-predictions">
        <h2>Sample Predictions</h2>
        <div id="sample-predictions">{nnSamplePredictions}</div>
      </div>

      <div id="nn-scatter-plot" className={styles.chart}></div>

      <div id="prediction-results">
        <h2>Prediction Results</h2>
        <div id="results">{predictionResults}</div>
      </div>

      <div id="prediction-chart" className={styles.chart}></div>

      <div id="chat-box">
        <h2>Chat with Assistant</h2>
        <div id="chat-messages">
          {chatMessages.map((msg, index) => (
            <div key={index} className={styles.chatMessage}>
              <span className={styles.messageSender}>{msg.sender}: </span>
              <span className={styles.messageContent}>{msg.content}</span>
            </div>
          ))}
        </div>
        <textarea id="chat-input" placeholder="Type your message here..."></textarea>
        <button id="send-message" onClick={handleSendMessage}>Send</button>
      </div>

      <div id="report-progress">
        <h2>Report Generation Progress</h2>
        <div id="progress">{reportProgress}%</div>
      </div>

      <div id="report-download">
        <h2>Download Report</h2>
        <a href={reportDownloadLink} id="download-link" className={styles.button}>Download Report</a>
      </div>
    </div>
  );
};

export default Home;

