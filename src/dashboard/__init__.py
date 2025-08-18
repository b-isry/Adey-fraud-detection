"""
Dashboard module for the Fraud Detection System.

Contains Streamlit-based interactive dashboard for monitoring and analysis.
"""

from .streamlit_app import create_dashboard
from .components import (
    SidebarComponent,
    MetricsComponent,
    ChartsComponent,
    AlertsComponent,
    ModelInfoComponent
)

__all__ = [
    "create_dashboard",
    "SidebarComponent",
    "MetricsComponent",
    "ChartsComponent", 
    "AlertsComponent",
    "ModelInfoComponent"
] 