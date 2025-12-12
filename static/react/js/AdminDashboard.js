import React, { useState, useEffect } from 'react';
import './AdminDashboard.css';

const AdminDashboard = () => {
    const [stats, setStats] = useState({
        totalUsers: 0,
        totalDetections: 0,
        accuracy: 0,
        uptime: 0
    });
    
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('overview');

    useEffect(() => {
        // Simulate loading data
        setTimeout(() => {
            setStats({
                totalUsers: 1247,
                totalDetections: 15689,
                accuracy: 99.2,
                uptime: 99.9
            });
            setDatasets([
                {
                    id: 1,
                    name: 'FaceForensics++',
                    size: '2.3GB',
                    uploadDate: '2024-01-15',
                    status: 'active'
                },
                {
                    id: 2,
                    name: 'DeepFake Detection Challenge',
                    size: '1.8GB',
                    uploadDate: '2024-01-10',
                    status: 'active'
                },
                {
                    id: 3,
                    name: 'DeeperForensics-1.0',
                    size: '3.1GB',
                    uploadDate: '2024-01-05',
                    status: 'pending'
                }
            ]);
            setLoading(false);
        }, 2000);
    }, []);

    const StatCard = ({ icon, number, label, change, changeType }) => (
        <div className="admin-stat-card">
            <div className="admin-stat-icon">
                <i className={icon}></i>
            </div>
            <div className="admin-stat-number">{number}</div>
            <div className="admin-stat-label">{label}</div>
            {change && (
                <div className={`admin-stat-change ${changeType}`}>
                    <i className={`fas fa-arrow-${changeType === 'positive' ? 'up' : 'down'}`}></i>
                    {change}
                </div>
            )}
        </div>
    );

    const DatasetItem = ({ dataset }) => (
        <div className="admin-dataset-item">
            <div className="admin-dataset-info">
                <h4>{dataset.name}</h4>
                <p>Size: {dataset.size}</p>
                <p>Uploaded: {dataset.uploadDate}</p>
            </div>
            <div className="admin-dataset-actions">
                <span className={`admin-status ${dataset.status}`}>
                    {dataset.status}
                </span>
                <button className="admin-btn admin-btn-secondary">
                    <i className="fas fa-eye"></i> View
                </button>
                <button className="admin-btn admin-btn-primary">
                    <i className="fas fa-download"></i> Download
                </button>
            </div>
        </div>
    );

    if (loading) {
        return (
            <div className="admin-loading">
                <div className="admin-loading-spinner"></div>
                <div className="admin-loading-text">Loading Dashboard...</div>
            </div>
        );
    }

    return (
        <div className="admin-container">
            {/* Sidebar */}
            <div className="admin-sidebar">
                <div className="admin-sidebar-header">
                    <a href="#" className="admin-sidebar-brand">
                        <i className="fas fa-shield-alt"></i>
                        Admin Panel
                    </a>
                </div>
                
                <nav className="admin-sidebar-nav">
                    <div className="admin-nav-section">
                        <div className="admin-nav-title">Main</div>
                        <div className="admin-nav-item">
                            <a href="#" 
                               className={`admin-nav-link ${activeTab === 'overview' ? 'active' : ''}`}
                               onClick={() => setActiveTab('overview')}>
                                <i className="fas fa-tachometer-alt"></i>
                                Overview
                            </a>
                        </div>
                        <div className="admin-nav-item">
                            <a href="#" 
                               className={`admin-nav-link ${activeTab === 'users' ? 'active' : ''}`}
                               onClick={() => setActiveTab('users')}>
                                <i className="fas fa-users"></i>
                                Users
                            </a>
                        </div>
                        <div className="admin-nav-item">
                            <a href="#" 
                               className={`admin-nav-link ${activeTab === 'datasets' ? 'active' : ''}`}
                               onClick={() => setActiveTab('datasets')}>
                                <i className="fas fa-database"></i>
                                Datasets
                            </a>
                        </div>
                    </div>
                    
                    <div className="admin-nav-section">
                        <div className="admin-nav-title">System</div>
                        <div className="admin-nav-item">
                            <a href="#" className="admin-nav-link">
                                <i className="fas fa-cog"></i>
                                Settings
                            </a>
                        </div>
                        <div className="admin-nav-item">
                            <a href="#" className="admin-nav-link">
                                <i className="fas fa-chart-line"></i>
                                Analytics
                            </a>
                        </div>
                        <div className="admin-nav-item">
                            <a href="#" className="admin-nav-link">
                                <i className="fas fa-shield-alt"></i>
                                Security
                            </a>
                        </div>
                    </div>
                </nav>
            </div>

            {/* Main Content */}
            <div className="admin-main">
                {/* Header */}
                <div className="admin-header">
                    <h1>Admin Dashboard</h1>
                    <div className="admin-header-actions">
                        <button className="admin-btn admin-btn-secondary">
                            <i className="fas fa-plus"></i> Add Dataset
                        </button>
                        <button className="admin-btn admin-btn-primary">
                            <i className="fas fa-download"></i> Export Report
                        </button>
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="admin-stats-grid">
                    <StatCard 
                        icon="fas fa-users"
                        number={stats.totalUsers.toLocaleString()}
                        label="Total Users"
                        change="+12%"
                        changeType="positive"
                    />
                    <StatCard 
                        icon="fas fa-search"
                        number={stats.totalDetections.toLocaleString()}
                        label="Total Detections"
                        change="+8%"
                        changeType="positive"
                    />
                    <StatCard 
                        icon="fas fa-bullseye"
                        number={`${stats.accuracy}%`}
                        label="Model Accuracy"
                        change="+2.1%"
                        changeType="positive"
                    />
                    <StatCard 
                        icon="fas fa-server"
                        number={`${stats.uptime}%`}
                        label="System Uptime"
                        change="+0.1%"
                        changeType="positive"
                    />
                </div>

                {/* Content Sections */}
                {activeTab === 'overview' && (
                    <>
                        <div className="admin-section">
                            <div className="admin-section-header">
                                <h2 className="admin-section-title">Recent Activity</h2>
                                <div className="admin-section-actions">
                                    <button className="admin-btn admin-btn-secondary">
                                        View All
                                    </button>
                                </div>
                            </div>
                            
                            <div className="admin-activity-list">
                                <div className="admin-activity-item">
                                    <div className="admin-activity-icon">
                                        <i className="fas fa-user-plus"></i>
                                    </div>
                                    <div className="admin-activity-content">
                                        <h4>New user registered</h4>
                                        <p>john.doe@example.com joined the platform</p>
                                        <span className="admin-activity-time">2 minutes ago</span>
                                    </div>
                                </div>
                                
                                <div className="admin-activity-item">
                                    <div className="admin-activity-icon">
                                        <i className="fas fa-search"></i>
                                    </div>
                                    <div className="admin-activity-content">
                                        <h4>Deepfake detection completed</h4>
                                        <p>Video analysis finished with 98.5% confidence</p>
                                        <span className="admin-activity-time">15 minutes ago</span>
                                    </div>
                                </div>
                                
                                <div className="admin-activity-item">
                                    <div className="admin-activity-icon">
                                        <i className="fas fa-database"></i>
                                    </div>
                                    <div className="admin-activity-content">
                                        <h4>Dataset uploaded</h4>
                                        <p>New training dataset "FaceForensics++ v2" added</p>
                                        <span className="admin-activity-time">1 hour ago</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </>
                )}

                {activeTab === 'datasets' && (
                    <div className="admin-section">
                        <div className="admin-section-header">
                            <h2 className="admin-section-title">Training Datasets</h2>
                            <div className="admin-section-actions">
                                <button className="admin-btn admin-btn-primary">
                                    <i className="fas fa-upload"></i> Upload New
                                </button>
                            </div>
                        </div>
                        
                        <div className="admin-datasets-list">
                            {datasets.map(dataset => (
                                <DatasetItem key={dataset.id} dataset={dataset} />
                            ))}
                        </div>
                    </div>
                )}

                {activeTab === 'users' && (
                    <div className="admin-section">
                        <div className="admin-section-header">
                            <h2 className="admin-section-title">User Management</h2>
                            <div className="admin-section-actions">
                                <button className="admin-btn admin-btn-primary">
                                    <i className="fas fa-user-plus"></i> Add User
                                </button>
                            </div>
                        </div>
                        
                        <table className="admin-table">
                            <thead>
                                <tr>
                                    <th>User ID</th>
                                    <th>Username</th>
                                    <th>Email</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>#001</td>
                                    <td>john_doe</td>
                                    <td>john@example.com</td>
                                    <td><span className="admin-status active">Active</span></td>
                                    <td>
                                        <button className="admin-btn admin-btn-secondary">
                                            <i className="fas fa-edit"></i>
                                        </button>
                                    </td>
                                </tr>
                                <tr>
                                    <td>#002</td>
                                    <td>jane_smith</td>
                                    <td>jane@example.com</td>
                                    <td><span className="admin-status active">Active</span></td>
                                    <td>
                                        <button className="admin-btn admin-btn-secondary">
                                            <i className="fas fa-edit"></i>
                                        </button>
                                    </td>
                                </tr>
                                <tr>
                                    <td>#003</td>
                                    <td>admin_user</td>
                                    <td>admin@example.com</td>
                                    <td><span className="admin-status active">Active</span></td>
                                    <td>
                                        <button className="admin-btn admin-btn-secondary">
                                            <i className="fas fa-edit"></i>
                                        </button>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AdminDashboard;
