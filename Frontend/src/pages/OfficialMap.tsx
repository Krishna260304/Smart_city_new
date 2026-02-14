import { useMemo, useState } from 'react';
import { OfficialDashboardLayout } from '@/components/layout/OfficialDashboardLayout';
import { LeafletMap } from '@/components/maps/LeafletMap'; // Import LeafletMap
import { useHeatmap, useIncidents, useTickets } from '@/hooks/use-data';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const OfficialMap = () => {
  const { incidents, loading: incidentsLoading } = useIncidents();
  const { tickets, loading: ticketsLoading } = useTickets();
  const { points: heatmapPoints } = useHeatmap();
  
  const [showIncidents, setShowIncidents] = useState(true);
  const [showTickets, setShowTickets] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);

  const markers = useMemo(() => {
    const items: {
      id: string;
      position: { lat: number; lng: number };
      title: string;
      description?: string;
      status?: string;
      priority?: string;
      type?: string;
    }[] = [];
    
    if (showIncidents) {
      incidents.forEach((i) => {
        if (typeof i.latitude === 'number' && typeof i.longitude === 'number') {
          items.push({
            id: `incident-${i.id}`,
            position: { lat: i.latitude, lng: i.longitude },
            title: i.title || 'Incident',
            description: i.description,
            status: i.status,
            priority: i.priority,
            type: 'incident'
          });
        }
      });
    }
    
    if (showTickets) {
      tickets.forEach((t) => {
        if (typeof t.latitude === 'number' && typeof t.longitude === 'number') {
          items.push({
            id: `ticket-${t.id}`,
            position: { lat: t.latitude, lng: t.longitude },
            title: t.title || 'Ticket',
            description: t.description,
            status: t.status,
            priority: t.priority,
            type: 'ticket'
          });
        }
      });
    }
    return items;
  }, [incidents, tickets, showIncidents, showTickets]);

  return (
    <OfficialDashboardLayout>
      <div className="space-y-6 animate-fade-in">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-heading font-bold text-foreground">
              Live Map
            </h1>
            <p className="text-muted-foreground">
              Real-time incident and ticket visibility across zones
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant={showIncidents ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowIncidents(v => !v)}
            >
              Incidents
            </Button>
            <Button
              variant={showTickets ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowTickets(v => !v)}
            >
              Tickets
            </Button>
            <Button
              variant={showHeatmap ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowHeatmap(v => !v)}
            >
              Heatmap
            </Button>
          </div>
        </div>

        <LeafletMap
          markers={markers}
          heatmapPoints={heatmapPoints}
          showHeatmap={showHeatmap}
          zoom={12}
          height="560px"
        />

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Incident List */}
          <div className="bg-card rounded-xl border border-border p-4">
            <h2 className="font-heading font-semibold text-foreground mb-3">
              Recent Incidents
            </h2>
            {incidentsLoading && (
              <div className="text-sm text-muted-foreground">Loading incidents...</div>
            )}
            {!incidentsLoading && incidents.length === 0 && (
              <div className="text-sm text-muted-foreground">No incidents found</div>
            )}
            <div className="space-y-3">
              {incidents.slice(0, 8).map((i) => (
                <div key={i.id} className="p-3 border border-border rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-foreground">{i.title}</div>
                    <div className={cn("text-xs px-2 py-0.5 rounded-full border", i.status === 'resolved' ? 'badge-success' : i.status === 'in_progress' ? 'badge-warning' : 'badge-info')}>
                      {i.status}
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {i.location}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Ticket List */}
          <div className="bg-card rounded-xl border border-border p-4">
            <h2 className="font-heading font-semibold text-foreground mb-3">
              Active Tickets
            </h2>
            {ticketsLoading && (
              <div className="text-sm text-muted-foreground">Loading tickets...</div>
            )}
            {!ticketsLoading && tickets.length === 0 && (
              <div className="text-sm text-muted-foreground">No tickets found</div>
            )}
            <div className="space-y-3">
              {tickets.slice(0, 8).map((t) => (
                <div key={t.id} className="p-3 border border-border rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-foreground">{t.title}</div>
                    <div className={cn("text-xs px-2 py-0.5 rounded-full border", t.status === 'resolved' ? 'badge-success' : t.status === 'in_progress' ? 'badge-warning' : 'badge-info')}>
                      {t.status}
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {t.location}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </OfficialDashboardLayout>
  );
};

export default OfficialMap;















// import { useMemo, useState } from 'react';
// import { OfficialDashboardLayout } from '@/components/layout/OfficialDashboardLayout';
// import { GoogleMap } from '@/components/maps/GoogleMap';
// import { useHeatmap, useIncidents, useTickets } from '@/hooks/use-data';
// import { Button } from '@/components/ui/button';
// import { cn } from '@/lib/utils';

// const OfficialMap = () => {
//   const { incidents, loading: incidentsLoading } = useIncidents();
//   const { tickets, loading: ticketsLoading } = useTickets();
//   const { points: heatmapPoints } = useHeatmap();
//   const [showIncidents, setShowIncidents] = useState(true);
//   const [showTickets, setShowTickets] = useState(true);
//   const [showHeatmap, setShowHeatmap] = useState(true);

//   const markers = useMemo(() => {
//     const items: {
//       id: string;
//       position: { lat: number; lng: number };
//       title: string;
//       description?: string;
//       status?: string;
//       priority?: string;
//       type?: string;
//     }[] = [];
//     if (showIncidents) {
//       incidents.forEach((i) => {
//         if (typeof i.latitude === 'number' && typeof i.longitude === 'number') {
//           items.push({
//             id: `incident-${i.id}`,
//             position: { lat: i.latitude, lng: i.longitude },
//             title: i.title || 'Incident',
//             description: i.description,
//             status: i.status,
//             priority: i.priority,
//             type: 'incident'
//           });
//         }
//       });
//     }
//     if (showTickets) {
//       tickets.forEach((t) => {
//         if (typeof t.latitude === 'number' && typeof t.longitude === 'number') {
//           items.push({
//             id: `ticket-${t.id}`,
//             position: { lat: t.latitude, lng: t.longitude },
//             title: t.title || 'Ticket',
//             description: t.description,
//             status: t.status,
//             priority: t.priority,
//             type: 'ticket'
//           });
//         }
//       });
//     }
//     return items;
//   }, [incidents, tickets, showIncidents, showTickets]);

//   return (
//     <OfficialDashboardLayout>
//       <div className="space-y-6 animate-fade-in">
//         <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
//           <div>
//             <h1 className="text-2xl font-heading font-bold text-foreground">
//               Live Map
//             </h1>
//             <p className="text-muted-foreground">
//               Real-time incident and ticket visibility across zones
//             </p>
//           </div>
//           <div className="flex items-center gap-2">
//             <Button
//               variant={showIncidents ? 'default' : 'outline'}
//               size="sm"
//               onClick={() => setShowIncidents(v => !v)}
//             >
//               Incidents
//             </Button>
//             <Button
//               variant={showTickets ? 'default' : 'outline'}
//               size="sm"
//               onClick={() => setShowTickets(v => !v)}
//             >
//               Tickets
//             </Button>
//             <Button
//               variant={showHeatmap ? 'default' : 'outline'}
//               size="sm"
//               onClick={() => setShowHeatmap(v => !v)}
//             >
//               Heatmap
//             </Button>
//           </div>
//         </div>

//         <GoogleMap
//           apiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY || ''}
//           markers={markers}
//           heatmapPoints={heatmapPoints}
//           showHeatmap={showHeatmap}
//           zoom={12}
//           height="560px"
//         />

//         <div className="grid lg:grid-cols-2 gap-6">
//           <div className="bg-card rounded-xl border border-border p-4">
//             <h2 className="font-heading font-semibold text-foreground mb-3">
//               Recent Incidents
//             </h2>
//             {incidentsLoading && (
//               <div className="text-sm text-muted-foreground">Loading incidents...</div>
//             )}
//             {!incidentsLoading && incidents.length === 0 && (
//               <div className="text-sm text-muted-foreground">No incidents found</div>
//             )}
//             <div className="space-y-3">
//               {incidents.slice(0, 8).map((i) => (
//                 <div key={i.id} className="p-3 border border-border rounded-lg">
//                   <div className="flex items-center justify-between">
//                     <div className="font-medium text-foreground">{i.title}</div>
//                     <div className={cn("text-xs px-2 py-0.5 rounded-full border", i.status === 'resolved' ? 'badge-success' : i.status === 'in_progress' ? 'badge-warning' : 'badge-info')}>
//                       {i.status}
//                     </div>
//                   </div>
//                   <div className="text-xs text-muted-foreground mt-1">
//                     {i.location}
//                   </div>
//                 </div>
//               ))}
//             </div>
//           </div>

//           <div className="bg-card rounded-xl border border-border p-4">
//             <h2 className="font-heading font-semibold text-foreground mb-3">
//               Active Tickets
//             </h2>
//             {ticketsLoading && (
//               <div className="text-sm text-muted-foreground">Loading tickets...</div>
//             )}
//             {!ticketsLoading && tickets.length === 0 && (
//               <div className="text-sm text-muted-foreground">No tickets found</div>
//             )}
//             <div className="space-y-3">
//               {tickets.slice(0, 8).map((t) => (
//                 <div key={t.id} className="p-3 border border-border rounded-lg">
//                   <div className="flex items-center justify-between">
//                     <div className="font-medium text-foreground">{t.title}</div>
//                     <div className={cn("text-xs px-2 py-0.5 rounded-full border", t.status === 'resolved' ? 'badge-success' : t.status === 'in_progress' ? 'badge-warning' : 'badge-info')}>
//                       {t.status}
//                     </div>
//                   </div>
//                   <div className="text-xs text-muted-foreground mt-1">
//                     {t.location}
//                   </div>
//                 </div>
//               ))}
//             </div>
//           </div>
//         </div>
//       </div>
//     </OfficialDashboardLayout>
//   );
// };

// export default OfficialMap;
