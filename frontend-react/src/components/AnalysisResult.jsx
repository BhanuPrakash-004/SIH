import React from 'react';

const AnalysisResult = ({ analysis }) => {
    if (!analysis) return null;

    const { summary, action_items, assigned_role } = analysis;

    return (
        <section className="bg-white p-8 rounded-xl shadow-2xl space-y-6">
            <h2 className="text-3xl font-bold text-gray-800">Document Analysis</h2>

            {/* Assigned Role */}
            <div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">👤 Suggested Assignee</h3>
                <div className="p-4 bg-blue-50 text-blue-800 rounded-lg border border-blue-200 font-semibold">
                    {assigned_role}
                </div>
                <p className="text-sm text-gray-500 mt-1">This task can be automatically routed via n8n.</p>
            </div>

            {/* Summary */}
            <div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">📄 Executive Summary</h3>
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <p className="text-gray-600 whitespace-pre-wrap">{summary}</p>
                </div>
            </div>

            {/* Action Items */}
            <div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">📌 Action Items</h3>
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <ul className="list-disc list-inside space-y-2 text-gray-700">
                        {action_items.map((item, index) => (
                            <li key={index}>{item}</li>
                        ))}
                    </ul>
                </div>
            </div>
        </section>
    );
};

export default AnalysisResult;