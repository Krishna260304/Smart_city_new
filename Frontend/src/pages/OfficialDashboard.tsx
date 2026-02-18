import { useEffect, useMemo, useState } from 'react';
import {
  ChevronDown,
  CheckCircle2,
  ClipboardList,
  Clock,
  Filter,
  RotateCcw,
  Users,
} from 'lucide-react';
import { OfficialDashboardLayout } from '@/components/layout/OfficialDashboardLayout';
import { SettingsModal } from '@/components/SettingsModal';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { cn } from '@/lib/utils';
import { useAnalyticsDashboard, useTickets } from '@/hooks/use-data';
import { authService, OfficialRole } from '@/services/auth';
import { ticketService, Ticket, TicketLogbookEntry } from '@/services/tickets';
import { usersService, WorkerOption } from '@/services/users';
import { useToast } from '@/hooks/use-toast';

const STATUS_BADGE: Record<string, string> = {
  open: 'badge-info',
  in_progress: 'badge-warning',
  resolved: 'badge-success',
  verified: 'badge-warning',
};

const ROLE_LABEL: Record<OfficialRole, string> = {
  department: 'Department',
  supervisor: 'Supervisor',
  field_inspector: 'Field Inspector',
  worker: 'Worker',
};

const normalizeStatus = (value?: string): string => {
  const status = (value || '').trim().toLowerCase();
  if (status === 'verified') {
    return 'in_progress';
  }
  return status;
};

const formatStatus = (value?: string): string => {
  const normalized = normalizeStatus(value);
  if (!normalized) {
    return 'UNKNOWN';
  }
  return normalized.replace('_', ' ').toUpperCase();
};

const toNumber = (value: unknown): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const formatDateTime = (value?: string): string => {
  if (!value) {
    return 'N/A';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return 'N/A';
  }
  return parsed.toLocaleString();
};

const toTitleCase = (value: string): string =>
  value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());

const toReadableText = (value: unknown): string => {
  if (value === null || value === undefined) {
    return '';
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => toReadableText(item))
      .filter(Boolean)
      .join(', ');
  }
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No';
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : '';
  }
  if (typeof value === 'string') {
    return value.trim();
  }
  if (typeof value === 'object') {
    const pairs = Object.entries(value as Record<string, unknown>)
      .map(([key, val]) => {
        const text = toReadableText(val);
        if (!text) {
          return '';
        }
        return `${toTitleCase(key)}: ${text}`;
      })
      .filter(Boolean);
    return pairs.join(', ');
  }
  return String(value);
};

const statusLabel = (value: unknown): string => {
  const raw = String(value || '').trim();
  if (!raw) {
    return '';
  }
  return toTitleCase(raw);
};

const formatLogbookDetails = (row: TicketLogbookEntry): string[] => {
  const action = String(row.action || '').trim().toLowerCase();
  const details = (row.details || {}) as Record<string, unknown>;
  const lines: string[] = [];

  if (action.startsWith('status_')) {
    const from = statusLabel(details.previousStatus);
    const to = statusLabel(details.newStatus || action.replace('status_', ''));
    if (from && to) {
      lines.push(`Status changed from ${from} to ${to}.`);
    } else if (to) {
      lines.push(`Status changed to ${to}.`);
    }
    const note = toReadableText(details.notes);
    if (note) {
      lines.push(`Note: ${note}`);
    }
    return lines;
  }

  if (action === 'assign_worker') {
    const workers = toReadableText(details.workerNames || details.workerName);
    if (workers) {
      lines.push(`Assigned workers: ${workers}`);
    }
    const specs = toReadableText(details.workerSpecializations || details.workerSpecialization);
    if (specs) {
      lines.push(`Specialization: ${specs}`);
    }
    const note = toReadableText(details.notes);
    if (note) {
      lines.push(`Note: ${note}`);
    }
    return lines;
  }

  if (action === 'field_progress_update') {
    const progress = toReadableText(details.progressPercent);
    const source = toReadableText(details.source);
    const confidence = toReadableText(details.confidence);
    const summary = toReadableText(details.summary);

    if (progress) {
      lines.push(`Progress updated to ${progress}%.`);
    }
    if (source || confidence) {
      lines.push(
        `Model source: ${source || 'N/A'}${confidence ? ` (confidence ${confidence})` : ''}`,
      );
    }
    if (summary) {
      lines.push(`Update: ${summary}`);
    }
    return lines;
  }

  Object.entries(details).forEach(([key, value]) => {
    const text = toReadableText(value);
    if (!text) {
      return;
    }
    lines.push(`${toTitleCase(key)}: ${text}`);
  });
  return lines;
};

const toIstDateKey = (value?: string): string => {
  const target = value ? new Date(value) : new Date();
  if (Number.isNaN(target.getTime())) {
    return '';
  }
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(target);
};

const normalizeLabel = (value: unknown): string =>
  String(value || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ');

const normalizePhone = (value: unknown): string => {
  const digits = String(value || '').replace(/\D/g, '');
  if (!digits) {
    return '';
  }
  return digits.length > 10 ? digits.slice(-10) : digits;
};

const uniqueNonEmpty = (values: Array<string | null | undefined>): string[] => {
  const output: string[] = [];
  const seen = new Set<string>();
  values.forEach((value) => {
    const current = String(value || '').trim();
    if (!current || seen.has(current)) {
      return;
    }
    seen.add(current);
    output.push(current);
  });
  return output;
};

const ticketAssignees = (ticket: Ticket) => {
  if (Array.isArray(ticket.assignees) && ticket.assignees.length > 0) {
    return ticket.assignees;
  }
  const legacyWorkerId = String(ticket.workerId || '').trim();
  if (!legacyWorkerId) {
    return [];
  }
  return [
    {
      workerId: legacyWorkerId,
      name: ticket.assigneeName || ticket.assignedTo || 'Worker',
      phone: ticket.assigneePhone,
      specialization: ticket.workerSpecialization,
      photoUrl: ticket.assigneePhotoUrl,
    },
  ];
};

const ticketWorkerIds = (ticket: Ticket): string[] =>
  uniqueNonEmpty([
    ticket.workerId,
    ...(ticket.workerIds || []),
    ...ticketAssignees(ticket).map((item) => item.workerId),
  ]);

const ticketAssigneeNames = (ticket: Ticket): string[] =>
  uniqueNonEmpty([
    ticket.assigneeName,
    ticket.assignedTo,
    ...(ticket.assigneeNames || []),
    ...ticketAssignees(ticket).map((item) => item.name),
  ]);

const ticketAssigneePhones = (ticket: Ticket): string[] =>
  uniqueNonEmpty([
    ticket.assigneePhone,
    ...(ticket.assigneePhones || []),
    ...ticketAssignees(ticket).map((item) => item.phone),
  ]).map((phone) => normalizePhone(phone));

const ticketWorkerSpecializations = (ticket: Ticket): string[] =>
  uniqueNonEmpty([
    ticket.workerSpecialization,
    ...(ticket.workerSpecializations || []),
    ...ticketAssignees(ticket).map((item) => item.specialization),
  ]);

const byRoleTickets = (role: OfficialRole, allTickets: Ticket[], user: Record<string, unknown> | null): Ticket[] => {
  if (role === 'department' || role === 'supervisor') {
    return allTickets;
  }

  if (role === 'field_inspector') {
    return allTickets.filter((ticket) => normalizeStatus(ticket.status) !== 'resolved');
  }

  const userIds = new Set(
    [user?.id, user?._id]
      .map((value) => String(value || '').trim())
      .filter(Boolean),
  );
  const userPhone = normalizePhone(user?.phone);
  const userLabels = new Set(
    [user?.name, user?.email, user?.phone].map((value) => normalizeLabel(value)).filter(Boolean),
  );
  if (userPhone) {
    userLabels.add(userPhone);
  }

  return allTickets.filter((ticket) => {
    const ids = ticketWorkerIds(ticket);
    if (ids.some((ticketWorkerId) => userIds.has(ticketWorkerId))) {
      return true;
    }

    const phones = ticketAssigneePhones(ticket);
    if (userPhone && phones.includes(userPhone)) {
      return true;
    }

    const assignees = ticketAssigneeNames(ticket)
      .map((value) => normalizeLabel(value))
      .filter(Boolean);
    return assignees.some((assignee) => userLabels.has(assignee));
  });
};

type OfficialViewMode = 'overview' | 'tickets';

interface OfficialDashboardProps {
  mode?: OfficialViewMode;
}

const OfficialDashboard = ({ mode = 'tickets' }: OfficialDashboardProps) => {
  const { toast } = useToast();
  const [showSettings, setShowSettings] = useState(false);
  const [query, setQuery] = useState('');
  const [workers, setWorkers] = useState<WorkerOption[]>([]);
  const [workersLoading, setWorkersLoading] = useState(false);
  const [assignmentDrafts, setAssignmentDrafts] = useState<Record<string, string[]>>({});
  const [progressDrafts, setProgressDrafts] = useState<Record<string, string>>({});
  const [submittingAssignId, setSubmittingAssignId] = useState<string | null>(null);
  const [submittingStatusId, setSubmittingStatusId] = useState<string | null>(null);
  const [submittingProgressId, setSubmittingProgressId] = useState<string | null>(null);
  const [selectedTicket, setSelectedTicket] = useState<Ticket | null>(null);
  const [logbookRows, setLogbookRows] = useState<TicketLogbookEntry[]>([]);
  const [logbookOpen, setLogbookOpen] = useState(false);
  const [logbookLoading, setLogbookLoading] = useState(false);

  const currentUser = authService.getCurrentUser() as Record<string, unknown> | null;
  const officialRole = (currentUser?.officialRole || '') as OfficialRole;
  const isDepartment = officialRole === 'department';
  const isSupervisor = officialRole === 'supervisor';
  const isFieldInspector = officialRole === 'field_inspector';
  const isWorker = officialRole === 'worker';
  const isOverviewMode = mode === 'overview';
  const canAssignAndVerify = isSupervisor || isDepartment;
  const canSubmitProgress = isFieldInspector;

  const {
    tickets,
    loading: ticketsLoading,
    error: ticketsError,
    refetch: refetchTickets,
  } = useTickets();
  const { data: analytics, refetch: refetchAnalytics } = useAnalyticsDashboard();

  useEffect(() => {
    const loadWorkers = async () => {
      if (!canAssignAndVerify || isOverviewMode) {
        setWorkers([]);
        return;
      }

      setWorkersLoading(true);
      const response = await usersService.listWorkers();
      if (response.success && response.data) {
        setWorkers(response.data);
      } else {
        setWorkers([]);
      }
      setWorkersLoading(false);
    };

    void loadWorkers();
  }, [canAssignAndVerify, isOverviewMode]);

  const visibleTickets = useMemo(() => {
    if (!officialRole) {
      return [];
    }

    const scoped = byRoleTickets(officialRole, tickets, currentUser);
    const term = query.trim().toLowerCase();
    if (!term) {
      return scoped;
    }

    return scoped.filter((ticket) =>
      [
        ticket.title,
        ticket.description,
        ticket.category,
        ticket.location,
        ticket.assignedTo,
        ticket.assigneeName,
        ticket.workerSpecialization,
        ...(ticket.assigneeNames || []),
        ...(ticket.workerSpecializations || []),
        ...ticketAssignees(ticket)
          .map((item) => [item.name, item.specialization].filter(Boolean).join(' '))
          .filter(Boolean),
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
        .includes(term),
    );
  }, [currentUser, officialRole, query, tickets]);

  const stats = useMemo(() => {
    const openCount = visibleTickets.filter((ticket) => normalizeStatus(ticket.status) === 'open').length;
    const inProgressCount = visibleTickets.filter(
      (ticket) => normalizeStatus(ticket.status) === 'in_progress',
    ).length;
    const resolvedCount = visibleTickets.filter(
      (ticket) => normalizeStatus(ticket.status) === 'resolved',
    ).length;

    return {
      total: visibleTickets.length,
      open: openCount,
      inProgress: inProgressCount,
      resolved: resolvedCount,
    };
  }, [visibleTickets]);

  const inspectorPendingDailyUpdates = useMemo(() => {
    if (!isFieldInspector) {
      return 0;
    }

    const todayIst = toIstDateKey();
    return visibleTickets.filter((ticket) => {
      if (normalizeStatus(ticket.status) !== 'in_progress') {
        return false;
      }

      const updatedToday = toIstDateKey(ticket.lastInspectorUpdateAt) === todayIst;
      return !updatedToday;
    }).length;
  }, [isFieldInspector, visibleTickets]);

  const refreshEverything = async () => {
    await Promise.all([refetchTickets(), refetchAnalytics()]);
  };

  const listLabel = isWorker ? 'Assigned Work' : isOverviewMode ? 'Tracked Tickets' : 'Visible Tickets';
  const listSearchPlaceholder = isWorker
    ? 'Search assigned work by title, category, place'
    : 'Search by title, category, location, assignee';
  const listEmptyState = isWorker ? 'No assigned work found.' : 'No tickets found for this role.';
  const overviewTickets = useMemo(() => visibleTickets.slice(0, 6), [visibleTickets]);
  const pageTitle = isOverviewMode
    ? `${ROLE_LABEL[officialRole]} Dashboard`
    : isWorker
      ? 'Assigned Work'
      : `${ROLE_LABEL[officialRole]} Tickets`;
  const pageDescription = isOverviewMode
    ? 'High-level metrics and recent activity snapshot.'
    : isDepartment
      ? 'Department can assign workers, verify, reopen/resolve tickets, and access read-only logbooks.'
      : isSupervisor
        ? 'Supervisor assigns one or more workers from the roster and verifies tickets.'
        : isFieldInspector
          ? 'Field inspector must submit daily progress before 6:00 PM IST. AI predicts completion (5%, 10%, ...).'
          : 'Worker dashboard shows only assigned work and live progress.';

  const handleAssignWorker = async (ticketId: string, selectedWorkerIds: string[]) => {
    const workerIds = uniqueNonEmpty(selectedWorkerIds);
    if (!workerIds.length) {
      toast({
        title: 'Workers Required',
        description: 'Select one or more workers.',
        variant: 'destructive',
      });
      return;
    }

    setSubmittingAssignId(ticketId);
    try {
      const response = await ticketService.assignTicket(ticketId, { workerIds });
      if (response.success) {
        toast({ title: 'Workers Assigned', description: 'Supervisor assignment saved.' });
        await refreshEverything();
      } else {
        toast({
          title: 'Assignment Failed',
          description: response.error || 'Could not assign worker.',
          variant: 'destructive',
        });
      }
    } finally {
      setSubmittingAssignId(null);
    }
  };

  const updateWorkerSelectionDraft = (
    ticketId: string,
    workerId: string,
    fallbackWorkerIds: string[],
    nextChecked: boolean,
  ) => {
    setAssignmentDrafts((prev) => {
      const current = new Set(prev[ticketId] ?? fallbackWorkerIds);
      if (nextChecked) {
        current.add(workerId);
      } else {
        current.delete(workerId);
      }
      return { ...prev, [ticketId]: Array.from(current) };
    });
  };

  const handleUpdateStatus = async (ticketId: string, nextStatus: 'open' | 'resolved' | 'verified') => {
    setSubmittingStatusId(ticketId);
    try {
      const response = await ticketService.updateStatus(ticketId, { status: nextStatus });
      if (response.success) {
        const statusMessage: Record<'open' | 'resolved' | 'verified', string> = {
          verified: 'Ticket verified and moved to IN PROGRESS.',
          open: 'Ticket reopened.',
          resolved: 'Ticket marked as resolved.',
        };
        toast({
          title: 'Status Updated',
          description: statusMessage[nextStatus],
        });
        await refreshEverything();
      } else {
        toast({
          title: 'Status Update Failed',
          description: response.error || 'Could not update ticket status.',
          variant: 'destructive',
        });
      }
    } finally {
      setSubmittingStatusId(null);
    }
  };

  const handleProgressUpdate = async (ticketId: string) => {
    const updateText = String(progressDrafts[ticketId] || '').trim();
    if (!updateText) {
      toast({
        title: 'Update Required',
        description: 'Enter progress details for AI estimation.',
        variant: 'destructive',
      });
      return;
    }

    setSubmittingProgressId(ticketId);
    try {
      const response = await ticketService.updateProgress(ticketId, { updateText });
      if (response.success) {
        toast({
          title: 'Progress Submitted',
          description: `Estimated completion: ${response.data?.progressPercent ?? 0}%.`,
        });
        setProgressDrafts((prev) => ({ ...prev, [ticketId]: '' }));
        await refreshEverything();
      } else {
        toast({
          title: 'Update Failed',
          description: response.error || 'Could not submit progress update.',
          variant: 'destructive',
        });
      }
    } finally {
      setSubmittingProgressId(null);
    }
  };

  const handleOpenLogbook = async (ticket: Ticket) => {
    if (!isDepartment) {
      return;
    }

    setSelectedTicket(ticket);
    setLogbookRows([]);
    setLogbookOpen(true);
    setLogbookLoading(true);

    const response = await ticketService.getLogbook(ticket.id);
    if (response.success && response.data) {
      setLogbookRows(response.data);
    } else {
      setLogbookRows([]);
    }

    setLogbookLoading(false);
  };

  if (!officialRole || !ROLE_LABEL[officialRole]) {
    return (
      <OfficialDashboardLayout>
        <div className="rounded-xl border border-border bg-card p-6 text-sm text-muted-foreground">
          This account does not have a valid official role.
        </div>
      </OfficialDashboardLayout>
    );
  }

  return (
    <>
      <SettingsModal open={showSettings} onOpenChange={setShowSettings} isOfficial />
      <OfficialDashboardLayout onSettingsClick={() => setShowSettings(true)}>
        <div className="space-y-6 animate-fade-in">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-2xl font-heading font-bold text-foreground">{pageTitle}</h1>
              <p className="text-muted-foreground">{pageDescription}</p>
            </div>
          </div>

          {ticketsError && (
            <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
              {ticketsError}
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="text-sm text-muted-foreground">{listLabel}</div>
              <div className="text-2xl font-heading font-bold text-foreground">{stats.total}</div>
            </div>
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="text-sm text-muted-foreground">Open</div>
              <div className="text-2xl font-heading font-bold text-info">{stats.open}</div>
            </div>
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="text-sm text-muted-foreground">In Progress</div>
              <div className="text-2xl font-heading font-bold text-warning">{stats.inProgress}</div>
            </div>
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="text-sm text-muted-foreground">
                {isFieldInspector ? 'Pending Daily Updates' : 'Resolved'}
              </div>
              <div className="text-2xl font-heading font-bold text-success">
                {isFieldInspector ? inspectorPendingDailyUpdates : stats.resolved}
              </div>
            </div>
          </div>

          {(isDepartment || isSupervisor) && (
            <div className="rounded-xl border border-border bg-card p-4 text-sm text-muted-foreground">
              AI dashboard totals: {toNumber(analytics?.tickets.total)} total, {toNumber(analytics?.tickets.open)} open,
              {` `}
              {toNumber(analytics?.tickets.inProgress)} in progress, {toNumber(analytics?.tickets.resolved)} resolved.
            </div>
          )}

          {isOverviewMode ? (
            <div className="rounded-xl border border-border bg-card p-4 space-y-4">
              <div>
                <h2 className="text-lg font-semibold text-foreground">Overview Snapshot</h2>
                <p className="text-sm text-muted-foreground">
                  High-level view only. Open the Tickets tab for assignment and workflow actions.
                </p>
              </div>

              {ticketsLoading && <div className="text-sm text-muted-foreground">Loading tickets...</div>}
              {!ticketsLoading && overviewTickets.length === 0 && (
                <div className="text-sm text-muted-foreground">{listEmptyState}</div>
              )}

              <div className="space-y-2">
                {overviewTickets.map((ticket) => {
                  const status = normalizeStatus(ticket.status);
                  const progressValue = ticket.progressPercent ?? 0;
                  const assignedWorkerNames = ticketAssigneeNames(ticket);

                  return (
                    <div key={ticket.id} className="rounded-lg border border-border p-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="text-sm font-medium text-foreground">{ticket.title}</div>
                        <span
                          className={cn(
                            'rounded-full border px-2 py-0.5 text-xs font-medium',
                            STATUS_BADGE[status] || 'badge-info',
                          )}
                        >
                          {formatStatus(status)}
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        Place: {ticket.location || 'N/A'} | Workers:{' '}
                        {assignedWorkerNames.length > 0 ? assignedWorkerNames.join(', ') : 'Not assigned'} | Completion:{' '}
                        {progressValue}%
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">Updated: {formatDateTime(ticket.updatedAt)}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-border bg-card p-4 space-y-4">
              <div className="relative">
                <Filter className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  className="pl-9"
                  placeholder={listSearchPlaceholder}
                />
              </div>

              {ticketsLoading && <div className="text-sm text-muted-foreground">Loading tickets...</div>}
              {!ticketsLoading && visibleTickets.length === 0 && (
                <div className="text-sm text-muted-foreground">{listEmptyState}</div>
              )}

              <div className="space-y-3">
                {visibleTickets.map((ticket) => {
                const status = normalizeStatus(ticket.status);
                const existingWorkerIds = ticketWorkerIds(ticket);
                const selectedWorkerIds = assignmentDrafts[ticket.id] ?? existingWorkerIds;
                const canVerify = status === 'open' && (isDepartment || selectedWorkerIds.length > 0);
                const isResolved = status === 'resolved';
                const supervisorResolveLocked = isSupervisor && Number(ticket.reopenCount || 0) > 0;
                  const progressValue = ticket.progressPercent ?? 0;
                  const assignedWorkerNames = ticketAssigneeNames(ticket);
                  const assignedWorkerSpecializations = ticketWorkerSpecializations(ticket);

                  return (
                    <div key={ticket.id} className="rounded-xl border border-border p-4">
                      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                        <div className="space-y-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="text-sm font-semibold text-foreground">{ticket.title}</span>
                            <span
                              className={cn(
                                'rounded-full border px-2 py-0.5 text-xs font-medium',
                                STATUS_BADGE[status] || 'badge-info',
                              )}
                            >
                              {formatStatus(status)}
                            </span>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {ticket.category} | {ticket.priority} priority
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Place: {ticket.location || 'N/A'}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Workers: {assignedWorkerNames.length > 0 ? assignedWorkerNames.join(', ') : 'Not assigned'}
                            {assignedWorkerSpecializations.length > 0
                              ? ` (${assignedWorkerSpecializations.join(', ')})`
                              : ''}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Completion: <span className="font-medium text-foreground">{progressValue}%</span>
                          </div>
                        </div>

                        <div className="text-xs text-muted-foreground">Updated: {formatDateTime(ticket.updatedAt)}</div>
                      </div>

                      {isDepartment && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {!isResolved ? (
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  disabled={submittingStatusId === ticket.id}
                                >
                                  <CheckCircle2 className="mr-1 h-4 w-4" />
                                  Update Status
                                  <ChevronDown className="ml-2 h-4 w-4 opacity-70" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="start">
                                <DropdownMenuItem
                                  onClick={() => void handleUpdateStatus(ticket.id, 'verified')}
                                  disabled={submittingStatusId === ticket.id || !canVerify}
                                >
                                  Verified
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={() => void handleUpdateStatus(ticket.id, 'resolved')}
                                  disabled={submittingStatusId === ticket.id}
                                >
                                  Resolved
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          ) : (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => void handleUpdateStatus(ticket.id, 'open')}
                              disabled={submittingStatusId === ticket.id}
                            >
                              <RotateCcw className="mr-1 h-4 w-4" />
                              Reopen
                            </Button>
                          )}

                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => void handleOpenLogbook(ticket)}
                          >
                            <ClipboardList className="mr-1 h-4 w-4" />
                            Read Logbook
                          </Button>
                        </div>
                      )}

                      {canAssignAndVerify && !isResolved && (
                        <div className="mt-3 grid gap-3 md:grid-cols-[1fr_auto_auto] md:items-center">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                type="button"
                                variant="outline"
                                className="w-full justify-between"
                                disabled={workersLoading && workers.length === 0}
                              >
                                <span className="truncate">
                                  {workersLoading
                                    ? 'Loading workers...'
                                    : selectedWorkerIds.length > 0
                                      ? `${selectedWorkerIds.length} worker${selectedWorkerIds.length > 1 ? 's' : ''} selected`
                                      : 'Select workers'}
                                </span>
                                <ChevronDown className="ml-2 h-4 w-4 opacity-70" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="start" className="w-[380px] max-h-72 overflow-y-auto">
                              <DropdownMenuLabel>Assign Workers</DropdownMenuLabel>
                              <DropdownMenuSeparator />
                              {!workersLoading && workers.length === 0 && (
                                <div className="px-2 py-2 text-xs text-muted-foreground">No workers available.</div>
                              )}
                              {!workersLoading &&
                                workers.map((worker) => {
                                  const checked = selectedWorkerIds.includes(worker.id);
                                  return (
                                    <DropdownMenuCheckboxItem
                                      key={worker.id}
                                      checked={checked}
                                      onCheckedChange={(nextValue) =>
                                        updateWorkerSelectionDraft(
                                          ticket.id,
                                          worker.id,
                                          existingWorkerIds,
                                          !!nextValue,
                                        )
                                      }
                                      onSelect={(event) => event.preventDefault()}
                                      disabled={submittingAssignId === ticket.id}
                                    >
                                      {worker.name}
                                      {worker.workerSpecialization ? ` - ${worker.workerSpecialization}` : ''}
                                    </DropdownMenuCheckboxItem>
                                  );
                                })}
                            </DropdownMenuContent>
                          </DropdownMenu>

                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => void handleAssignWorker(ticket.id, selectedWorkerIds)}
                            disabled={submittingAssignId === ticket.id || selectedWorkerIds.length === 0}
                          >
                            <Users className="mr-1 h-4 w-4" />
                            Assign Team
                          </Button>

                          {isSupervisor && (
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button size="sm" disabled={submittingStatusId === ticket.id}>
                                  <CheckCircle2 className="mr-1 h-4 w-4" />
                                  Update Status
                                  <ChevronDown className="ml-2 h-4 w-4 opacity-70" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem
                                  onClick={() => void handleUpdateStatus(ticket.id, 'verified')}
                                  disabled={submittingStatusId === ticket.id || !canVerify}
                                >
                                  Verified
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={() => void handleUpdateStatus(ticket.id, 'resolved')}
                                  disabled={submittingStatusId === ticket.id || supervisorResolveLocked}
                                >
                                  Resolved
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          )}

                          {isSupervisor && supervisorResolveLocked && (
                            <p className="text-xs text-muted-foreground md:col-span-3">
                              This ticket was reopened. Only department can mark it as resolved now.
                            </p>
                          )}
                        </div>
                      )}

                      {canSubmitProgress && !isResolved && (
                        <div className="mt-3 space-y-2">
                          <Textarea
                            rows={3}
                            placeholder="Daily field update before 6:00 PM IST (e.g. trenching completed, fittings installed, testing started)."
                            value={progressDrafts[ticket.id] || ''}
                            onChange={(event) => {
                              const value = event.target.value;
                              setProgressDrafts((prev) => ({ ...prev, [ticket.id]: value }));
                            }}
                          />
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-xs text-muted-foreground">
                              Last inspector update: {formatDateTime(ticket.lastInspectorUpdateAt)}
                            </p>
                            <Button
                              size="sm"
                              onClick={() => void handleProgressUpdate(ticket.id)}
                              disabled={submittingProgressId === ticket.id}
                            >
                              <Clock className="mr-1 h-4 w-4" />
                              Submit Daily Update
                            </Button>
                          </div>
                        </div>
                      )}

                      {isWorker && (
                        <div className="mt-3 rounded-md border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
                          <div className="font-medium text-foreground">Assigned Work Details</div>
                          <div className="mt-1">Supervisor: {ticket.assignedBySupervisorId || 'N/A'}</div>
                          <div>Verified At: {formatDateTime(ticket.verifiedAt)}</div>
                          <div>Progress Summary: {ticket.progressSummary || 'No inspector note yet.'}</div>
                        </div>
                      )}

                      {isSupervisor && isResolved && (
                        <div className="mt-3 rounded-md border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
                          Resolved ticket. If reopening is required, department must reopen first.
                        </div>
                      )}

                      {isFieldInspector && isResolved && (
                        <div className="mt-3 rounded-md border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
                          This ticket is resolved. Daily update is no longer required.
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

        </div>
      </OfficialDashboardLayout>

      <Dialog open={logbookOpen} onOpenChange={setLogbookOpen}>
        <DialogContent className="sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>Read-Only Department Logbook</DialogTitle>
            <DialogDescription>
              Immutable timeline for ticket {selectedTicket?.id || ''}. Only department users can access this view.
            </DialogDescription>
          </DialogHeader>

          {logbookLoading && <div className="text-sm text-muted-foreground">Loading logbook...</div>}

          {!logbookLoading && logbookRows.length === 0 && (
            <div className="text-sm text-muted-foreground">No logbook entries found.</div>
          )}

          <div className="max-h-[60vh] space-y-2 overflow-y-auto pr-1">
            {logbookRows.map((row) => {
              const detailLines = formatLogbookDetails(row);
              return (
                <div key={row.id} className="rounded-md border border-border p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="text-sm font-semibold text-foreground">
                      {String(row.action || '').replace(/_/g, ' ').toUpperCase()}
                    </div>
                    <div className="text-xs text-muted-foreground">{formatDateTime(row.createdAt)}</div>
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    Actor: {row.actorName || row.actorUserId || 'Unknown'}
                    {row.actorOfficialRole ? ` (${row.actorOfficialRole})` : ''}
                  </div>
                  <div className="mt-2 rounded bg-muted/40 p-2 text-xs text-foreground">
                    {detailLines.length > 0 ? (
                      detailLines.map((line, index) => <div key={`${row.id}-${index}`}>{line}</div>)
                    ) : (
                      'No additional details'
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default OfficialDashboard;
