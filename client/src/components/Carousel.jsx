import React, { useState, useEffect } from 'react';
import logo_icon from '../assets/icons/logo.png';
import group_icon from '../assets/icons/Group.png';
import group2_icon from '../assets/icons/Group2.png';
import group3_icon from '../assets/icons/Group3.png';
import group8_icon from '../assets/icons/Group3.1.png';

export default function Carousel() {
  const [currentPage, setCurrentPage] = useState(0);

  // Define pages with content and (optional) background colors.
  const pages = [
    {
      content: (
        <>
          {/* Logo Container */}
          <div className="mb-4 w-[150px] h-[55px]">
            <img
              src={logo_icon}
              alt="UST Logo"
              className="w-full h-full object-contain"
              loading="eager"
            />
          </div>
          <div>
            <h2 className="text-2xl font-semibold mb-4">
              Provide{' '}
              <span className="bg-teal-100 px-1">
                Project Description
              </span>
            </h2>
            <div className="relative flex justify-center items-center mt-5">
              <img
                src={group_icon}
                alt="Group"
                className="w-[368px]"
              />
            </div>
          </div>
        </>
      ),
      backgroundColor: '#E2F9FD', // Optional; you can also use Tailwind classes if configured.
    },
    {
      content: (
        <>
          <div className="mb-4 w-[150px] h-[55px]">
            <img
              src={logo_icon}
              alt="UST Logo"
              className="w-full h-full object-contain"
              loading="eager"
            />
          </div>
          <div>
            <h2 className="text-2xl font-semibold mb-4">
              Provide{' '}
              <span className="bg-rose-100 px-1">
                Image
              </span>
            </h2>
            <div className="relative flex justify-center items-center mt-5">
              <img
                src={group2_icon}
                alt="Group"
                className="w-[368px]"
              />
            </div>
          </div>
        </>
      ),
      backgroundColor: '#E2F9FD',
    },
    {
      content: (
        <>
          <div className="mb-4 w-[150px] h-[55px]">
            <img
              src={logo_icon}
              alt="UST Logo"
              className="w-full h-full object-contain"
              loading="eager"
            />
          </div>
          <div>
            <h2 className="text-2xl font-semibold mb-4">
              Provide{' '}
              <span className="bg-yellow-100 px-1">
                Meeting Audios
              </span>
            </h2>
            <div className="relative flex justify-center items-center mt-5">
              <img
                src={group3_icon}
                alt="Group"
                className="w-[352px]"
              />
              <img
                src={group8_icon}
                alt="Detail Icon"
                className="absolute top-0 right-[-70px] w-[40px]"
              />
            </div>
          </div>
        </>
      ),
      backgroundColor: '#E2F9FD',
    },
  ];

  // Cycle through pages every 3 seconds.
  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentPage((prevPage) =>
        prevPage === pages.length - 1 ? 0 : prevPage + 1
      );
    }, 3000);
    return () => clearInterval(intervalId);
  }, [pages.length]);

  return (
    <div
      className="h-screen flex items-center justify-center"
      style={{ backgroundColor: pages[currentPage].backgroundColor }}
    >
      <div className="text-center p-8">
        {pages[currentPage].content}
      </div>
    </div>
  );
}