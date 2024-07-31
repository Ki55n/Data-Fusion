import React from 'react';
interface NavItemProps {
  href: string;
  isActive: boolean;
  children: React.ReactNode;
}
export default function NavItem({ href, isActive, children }: NavItemProps) {
    return (
        <li>
            <a
                href={href}
                className={`block py-4 text-base rounded hover:bg-tertiary hover-text-shadow ${isActive ? 'bg-primary text-white' : 'bg-slate-50'} nav-item`}
            >
                {children}
            </a>
        </li>
    )
}
