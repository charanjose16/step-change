import React from 'react';
import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-6xl font-bold text-gray-800">404</h1>
      <p className="mt-4 text-xl text-gray-600">Page Not Found</p>
      <Link 
        to="/"
        className="mt-6 bg-teal-600 text-white px-4 py-2 rounded-lg transition hover:bg-teal-700"
      >
        Go Home
      </Link>
    </div>
  );
}