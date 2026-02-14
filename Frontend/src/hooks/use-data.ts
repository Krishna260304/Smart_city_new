import { useState, useEffect } from 'react';
import { incidentService, Incident, IncidentStats } from '@/services/incidents';
import { ticketService, Ticket, TicketStats } from '@/services/tickets';
import { API_CONFIG } from '@/config/api';
import { analyticsService, AnalyticsDashboard, HeatmapPoint, TrendPoint } from '@/services/analytics';
import { publicService, PublicSummary } from '@/services/public';

const hasAuthToken = () => !!localStorage.getItem('auth_token');

export const useIncidents = () => {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchIncidents = async () => {
    if (!hasAuthToken()) {
      setIncidents([]);
      setLoading(false);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    const response = await incidentService.getIncidents();
    
    if (response.success && response.data) {
      setIncidents(response.data);
    } else {
      setError(response.error || 'Failed to fetch incidents');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchIncidents();
  }, []);

  useEffect(() => {
    if (!hasAuthToken()) {
      return;
    }
    
    // Construct WebSocket URL - always use localhost:8000 for backend
    const wsUrl = import.meta.env.DEV 
      ? 'ws://localhost:8000/ws/incidents'
      : 'wss://127.0.0.1:8000/ws/incidents';
    
    const socket = new WebSocket(wsUrl);
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload?.type === 'NEW_INCIDENT' && payload.data) {
          setIncidents(prev => {
            const exists = prev.find(i => i.id === payload.data.id);
            if (exists) {
              return prev.map(i => i.id === payload.data.id ? payload.data : i);
            }
            return [payload.data, ...prev];
          });
        }
      } catch {
        return;
      }
    };
    return () => {
      socket.close();
    };
  }, []);

  return { incidents, loading, error, refetch: fetchIncidents };
};

export const useIncidentStats = () => {
  const [stats, setStats] = useState<IncidentStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    if (!hasAuthToken()) {
      setStats(null);
      setLoading(false);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    const response = await incidentService.getStats();
    
    if (response.success && response.data) {
      setStats(response.data);
    } else {
      setError(response.error || 'Failed to fetch stats');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchStats();
  }, []);

  return { stats, loading, error, refetch: fetchStats };
};

export const useTickets = (filters?: { status?: string; priority?: string; category?: string }) => {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTickets = async () => {
    if (!hasAuthToken()) {
      setTickets([]);
      setLoading(false);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    const response = await ticketService.getTickets(filters);
    
    if (response.success && response.data) {
      setTickets(response.data);
    } else {
      setError(response.error || 'Failed to fetch tickets');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchTickets();
  }, [filters?.status, filters?.priority, filters?.category]);

  return { tickets, loading, error, refetch: fetchTickets };
};

export const useTicketStats = () => {
  const [stats, setStats] = useState<TicketStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    if (!hasAuthToken()) {
      setStats(null);
      setLoading(false);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    const response = await ticketService.getStats();
    
    if (response.success && response.data) {
      setStats(response.data);
    } else {
      setError(response.error || 'Failed to fetch stats');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchStats();
  }, []);

  return { stats, loading, error, refetch: fetchStats };
};

export const useAnalyticsDashboard = () => {
  const [data, setData] = useState<AnalyticsDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboard = async () => {
    if (!hasAuthToken()) {
      setData(null);
      setLoading(false);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    const response = await analyticsService.getDashboard();
    if (response.success && response.data) {
      setData(response.data);
    } else {
      setError(response.error || 'Failed to fetch analytics');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchDashboard();
  }, []);

  return { data, loading, error, refetch: fetchDashboard };
};

export const useHeatmap = () => {
  const [points, setPoints] = useState<HeatmapPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHeatmap = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyticsService.getHeatmapData(); 
      setPoints(data);
    } catch (err) {
      console.error('Failed to load heatmap:', err);
      setError('Failed to fetch heatmap');
      setPoints([
        { lat: 20.5937, lng: 78.9629, weight: 1.0 },
        { lat: 20.6000, lng: 78.9700, weight: 0.8 },
        { lat: 20.5800, lng: 78.9500, weight: 0.6 },
      ]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchHeatmap();
  }, []);

  return { points, loading, error, refetch: fetchHeatmap };
};

export const useTrends = (days = 14) => {
  const [data, setData] = useState<TrendPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTrends = async () => {
    if (!hasAuthToken()) {
      setData([]);
      setLoading(false);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    const response = await analyticsService.getTrends(days);
    if (response.success && response.data) {
      setData(response.data);
    } else {
      setError(response.error || 'Failed to fetch trends');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchTrends();
  }, [days]);

  return { data, loading, error, refetch: fetchTrends };
};

export const usePublicSummary = () => {
  const [data, setData] = useState<PublicSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSummary = async () => {
    setLoading(true);
    setError(null);
    const response = await publicService.getSummary();
    if (response.success && response.data) {
      setData(response.data);
    } else {
      setError(response.error || 'Failed to fetch summary');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchSummary();
  }, []);

  return { data, loading, error, refetch: fetchSummary };
};





// import { useState, useEffect } from 'react';
// import { incidentService, Incident, IncidentStats } from '@/services/incidents';
// import { ticketService, Ticket, TicketStats } from '@/services/tickets';
// import { API_CONFIG } from '@/config/api';
// import { analyticsService, AnalyticsDashboard, HeatmapPoint, TrendPoint } from '@/services/analytics';
// import { publicService, PublicSummary } from '@/services/public';

// const hasAuthToken = () => !!localStorage.getItem('auth_token');

// export const useIncidents = () => {
//   const [incidents, setIncidents] = useState<Incident[]>([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   const fetchIncidents = async () => {
//     if (!hasAuthToken()) {
//       setIncidents([]);
//       setLoading(false);
//       setError(null);
//       return;
//     }
//     setLoading(true);
//     setError(null);
//     const response = await incidentService.getIncidents();
    
//     if (response.success && response.data) {
//       setIncidents(response.data);
//     } else {
//       setError(response.error || 'Failed to fetch incidents');
//     }
//     setLoading(false);
//   };

//   useEffect(() => {
//     fetchIncidents();
//   }, []);

//   useEffect(() => {
//     if (!hasAuthToken()) {
//       return;
//     }
//     const base = API_CONFIG.BASE_URL.replace(/^http/, 'ws').replace(/\/api\/?$/, '');
//     const socket = new WebSocket(`${base}/ws/incidents`);
//     socket.onmessage = (event) => {
//       try {
//         const payload = JSON.parse(event.data);
//         if (payload?.type === 'NEW_INCIDENT' && payload.data) {
//           setIncidents(prev => {
//             const exists = prev.find(i => i.id === payload.data.id);
//             if (exists) {
//               return prev.map(i => i.id === payload.data.id ? payload.data : i);
//             }
//             return [payload.data, ...prev];
//           });
//         }
//       } catch {
//         return;
//       }
//     };
//     return () => {
//       socket.close();
//     };
//   }, []);

//   return { incidents, loading, error, refetch: fetchIncidents };
// };



// export const useIncidentStats = () => {
//   const [stats, setStats] = useState<IncidentStats | null>(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   const fetchStats = async () => {
//     if (!hasAuthToken()) {
//       setStats(null);
//       setLoading(false);
//       setError(null);
//       return;
//     }
//     setLoading(true);
//     setError(null);
//     const response = await incidentService.getStats();
    
//     if (response.success && response.data) {
//       setStats(response.data);
//     } else {
//       setError(response.error || 'Failed to fetch stats');
//     }
//     setLoading(false);
//   };

//   useEffect(() => {
//     fetchStats();
//   }, []);

//   return { stats, loading, error, refetch: fetchStats };
// };



// export const useTickets = (filters?: { status?: string; priority?: string; category?: string }) => {
//   const [tickets, setTickets] = useState<Ticket[]>([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   const fetchTickets = async () => {
//     if (!hasAuthToken()) {
//       setTickets([]);
//       setLoading(false);
//       setError(null);
//       return;
//     }
//     setLoading(true);
//     setError(null);
//     const response = await ticketService.getTickets(filters);
    
//     if (response.success && response.data) {
//       setTickets(response.data);
//     } else {
//       setError(response.error || 'Failed to fetch tickets');
//     }
//     setLoading(false);
//   };

//   useEffect(() => {
//     fetchTickets();
//   }, [filters?.status, filters?.priority, filters?.category]);

//   return { tickets, loading, error, refetch: fetchTickets };
// };



// export const useTicketStats = () => {
//   const [stats, setStats] = useState<TicketStats | null>(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   const fetchStats = async () => {
//     if (!hasAuthToken()) {
//       setStats(null);
//       setLoading(false);
//       setError(null);
//       return;
//     }
//     setLoading(true);
//     setError(null);
//     const response = await ticketService.getStats();
    
//     if (response.success && response.data) {
//       setStats(response.data);
//     } else {
//       setError(response.error || 'Failed to fetch stats');
//     }
//     setLoading(false);
//   };

//   useEffect(() => {
//     fetchStats();
//   }, []);

//   return { stats, loading, error, refetch: fetchStats };
// };

// export const useAnalyticsDashboard = () => {
//   const [data, setData] = useState<AnalyticsDashboard | null>(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   const fetchDashboard = async () => {
//     if (!hasAuthToken()) {
//       setData(null);
//       setLoading(false);
//       setError(null);
//       return;
//     }
//     setLoading(true);
//     setError(null);
//     const response = await analyticsService.getDashboard();
//     if (response.success && response.data) {
//       setData(response.data);
//     } else {
//       setError(response.error || 'Failed to fetch analytics');
//     }
//     setLoading(false);
//   };

//   useEffect(() => {
//     fetchDashboard();
//   }, []);

//   return { data, loading, error, refetch: fetchDashboard };
// };

// export const useHeatmap = () => {
//   const [points, setPoints] = useState<HeatmapPoint[]>([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   const fetchHeatmap = async () => {
//     if (!hasAuthToken()) {
//       setPoints([]);
//       setLoading(false);
//       setError(null);
//       return;
//     }
//     setLoading(true);
//     setError(null);
//     const response = await analyticsService.getHeatmap();
//     if (response.success && response.data) {
//       setPoints(response.data);
//     } else {
//       setError(response.error || 'Failed to fetch heatmap');
//     }
//     setLoading(false);
//   };

//   useEffect(() => {
//     fetchHeatmap();
//   }, []);

//   return { points, loading, error, refetch: fetchHeatmap };
// };

// export const useTrends = (days = 14) => {
//   const [data, setData] = useState<TrendPoint[]>([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   const fetchTrends = async () => {
//     if (!hasAuthToken()) {
//       setData([]);
//       setLoading(false);
//       setError(null);
//       return;
//     }
//     setLoading(true);
//     setError(null);
//     const response = await analyticsService.getTrends(days);
//     if (response.success && response.data) {
//       setData(response.data);
//     } else {
//       setError(response.error || 'Failed to fetch trends');
//     }
//     setLoading(false);
//   };

//   useEffect(() => {
//     fetchTrends();
//   }, [days]);

//   return { data, loading, error, refetch: fetchTrends };
// };

// export const usePublicSummary = () => {
//   const [data, setData] = useState<PublicSummary | null>(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   const fetchSummary = async () => {
//     setLoading(true);
//     setError(null);
//     const response = await publicService.getSummary();
//     if (response.success && response.data) {
//       setData(response.data);
//     } else {
//       setError(response.error || 'Failed to fetch summary');
//     }
//     setLoading(false);
//   };

//   useEffect(() => {
//     fetchSummary();
//   }, []);

//   return { data, loading, error, refetch: fetchSummary };
// };
