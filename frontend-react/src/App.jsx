import React, { useState } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
import AnalysisResult from './components/AnalysisResult';
import Qa from './components/Qa';

// ## CHANGE ##: Set the correct base URL for your FastAPI backend
const API_BASE_URL = 'http://localhost:8000';

const App = () => {
    const [analysis, setAnalysis] = useState(null);
    const [documentName, setDocumentName] = useState('');
    
    // ## CHANGE ##: This will now hold the entire object { answer, sources }
    const [qaResponse, setQaResponse] = useState(null);
    
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleFileUpload = async (file) => {
        if (!file) return;
        setIsLoading(true);
        setError('');
        setAnalysis(null);
        setDocumentName('');
        setQaResponse(null); // Clear previous chat results on new upload

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post(`${API_BASE_URL}/api/upload-and-process`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setAnalysis(response.data);
            setDocumentName(file.name);
        } catch (err) {
            const errorMsg = err.response?.data?.detail || 'Error uploading or processing file.';
            setError(errorMsg);
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleQaSubmit = async (query) => {
        if (!query) return;
        setIsLoading(true);
        setError('');
        try {
            const response = await axios.post(`${API_BASE_URL}/api/chat`, { query });
            // ## CHANGE ##: Store the entire response object, not just the answer
            setQaResponse(response.data);
        } catch (err) {
            const errorMsg = err.response?.data?.detail || 'Error getting answer from knowledge base.';
            setError(errorMsg);
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 font-sans">
            <header className="bg-gradient-to-r from-orange-600 to-blue-700 text-white shadow-lg">
                <div className="container mx-auto px-6 py-8">
                    <h1 className="text-5xl font-extrabold tracking-tight">🧠 KMRL InsightEngine</h1>
                    <p className="mt-3 text-xl text-indigo-200">Your organization's centralized knowledge base, powered by AI.</p>
                </div>
            </header>

            <main className="container mx-auto px-6 py-12">
                {error && (
                    <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-8 rounded-md" role="alert">
                        <p className="font-bold">Error</p>
                        <p>{error}</p>
                    </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                    {/* Left Column: Upload and Analysis */}
                    <div className="space-y-8">
                        <section className="bg-white p-8 rounded-xl shadow-2xl">
                             <div className="flex items-center mb-6">
                                 <div className="bg-indigo-500 text-white rounded-full h-12 w-12 flex items-center justify-center text-2xl font-bold">1</div>
                                 <h2 className="text-3xl font-bold ml-4 text-gray-800">Add to Knowledge Base</h2>
                             </div>
                            <FileUpload onFileUpload={handleFileUpload} isLoading={isLoading} />
                            {documentName && !isLoading && (
                                <p className="mt-6 text-green-600 font-semibold bg-green-100 p-3 rounded-lg">
                                    ✅ Successfully processed and added to knowledge base: {documentName}
                                </p>
                            )}
                        </section>
                        
                        {isLoading && !analysis && <div className="text-center p-4">Analyzing document, please wait...</div>}
                        {analysis && <AnalysisResult analysis={analysis} />}
                    </div>

                    {/* Right Column: Chat with all documents */}
                    <section className="bg-white p-8 rounded-xl shadow-2xl">
                        <div className="flex items-center mb-6">
                            <div className="bg-indigo-500 text-white rounded-full h-12 w-12 flex items-center justify-center text-2xl font-bold">2</div>
                            <h2 className="text-3xl font-bold ml-4 text-gray-800">Chat with Your Data</h2>
                        </div>
                        <p className="text-gray-600 mb-6">Ask questions about any document you've uploaded. The AI will search the entire knowledge base to find the answer.</p>
                        <Qa onSubmit={handleQaSubmit} isLoading={isLoading} response={qaResponse} />
                    </section>
                </div>
            </main>
        </div>
    );
};

export default App;