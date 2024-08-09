import React from 'react';
interface NavigationProps {
  children: React.ReactNode;
}
export default function Navigation({ children }: NavigationProps) {
    return (
        <nav>
            <ul className="flex justify-between py-4 gap-3">
                {children}
            </ul>
        </nav>
    )
}
