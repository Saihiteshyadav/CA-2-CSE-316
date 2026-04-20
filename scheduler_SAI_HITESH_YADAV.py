"""
Energy-Efficient CPU Scheduling Algorithm
CSE-316 CA2 Project | Roll No: 35 | Krishan Kumar (R224EIB35) | Reg: 12410760

Implements an Energy-Efficient CPU Scheduler that combines:
- Dynamic Voltage and Frequency Scaling (DVFS)
- Power-aware Round Robin with adaptive time quantum
- EDF (Earliest Deadline First) for real-time tasks
- Sleep/idle state management
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import time
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import copy

# ─────────────────────────────────────────────
#  Core Data Structures
# ─────────────────────────────────────────────

class Process:
    def __init__(self, pid, name, arrival, burst, priority=1, deadline=None, power_class='normal'):
        self.pid = pid
        self.name = name
        self.arrival = arrival
        self.burst = burst
        self.remaining = burst
        self.priority = priority          # 1=low, 2=medium, 3=high
        self.deadline = deadline if deadline else arrival + burst * 3
        self.power_class = power_class    # 'low', 'normal', 'high'
        self.start_time = None
        self.finish_time = None
        self.waiting_time = 0
        self.turnaround_time = 0
        self.energy_consumed = 0.0
        self.frequency_used = []          # track freq per time unit

    def __repr__(self):
        return f"P{self.pid}({self.name})"


class CPUState:
    ACTIVE   = 'active'
    IDLE     = 'idle'
    SLEEP    = 'sleep'


class DVFSModel:
    """
    Simplified DVFS: voltage & power scale roughly as freq^2.
    Frequencies (normalized): 0.25, 0.50, 0.75, 1.0
    """
    LEVELS = [0.25, 0.50, 0.75, 1.0]
    # Power ~ V^2 * f, V scales ~linearly with f  =>  power ~ f^3 roughly
    POWER  = {0.25: 0.06, 0.50: 0.20, 0.75: 0.50, 1.0: 1.00}  # relative units

    @staticmethod
    def choose_freq(remaining_time, deadline, current_time):
        """Pick the lowest frequency that can finish before deadline."""
        slack = deadline - current_time
        if slack <= 0:
            return 1.0   # must rush
        ratio = remaining_time / slack
        if ratio <= 0.25:
            return 0.25
        elif ratio <= 0.50:
            return 0.50
        elif ratio <= 0.75:
            return 0.75
        else:
            return 1.0

    @staticmethod
    def exec_time_at_freq(burst_units, freq):
        """Real-time units at given frequency (lower freq = slower)."""
        return burst_units / freq

    @staticmethod
    def energy(burst_units, freq):
        """Energy = power * time."""
        t = DVFSModel.exec_time_at_freq(burst_units, freq)
        p = DVFSModel.POWER[freq]
        return p * t


class EnergyEfficientScheduler:
    """
    Hybrid scheduler:
      1. Real-time tasks -> EDF with DVFS
      2. Normal tasks   -> Power-aware Round Robin (adaptive quantum)
      3. CPU idles when queue empty -> sleep state (0 energy draw)
    """

    def __init__(self, processes, base_quantum=4):
        self.processes   = sorted(copy.deepcopy(processes), key=lambda p: p.arrival)
        self.base_quantum = base_quantum
        self.timeline    = []   # list of (pid, start, end, freq, state)
        self.current_time = 0
        self.cpu_state   = CPUState.IDLE
        self.total_energy = 0.0
        self.idle_energy  = 0.005  # per unit time in idle
        self.sleep_energy = 0.001  # per unit time in sleep

    def _adaptive_quantum(self, process):
        """Scale quantum: high-priority / deadline-critical tasks get longer slice."""
        q = self.base_quantum
        if process.priority == 3:
            q = int(q * 1.5)
        elif process.priority == 1:
            q = max(2, int(q * 0.75))
        return q

    def run(self):
        remaining  = self.processes[:]
        ready_q    = deque()
        t = 0

        while remaining or ready_q:
            # Enqueue arrived processes
            arrived = [p for p in remaining if p.arrival <= t]
            for p in arrived:
                ready_q.append(p)
                remaining.remove(p)

            if not ready_q:
                # CPU sleep until next arrival
                next_arr = min(p.arrival for p in remaining) if remaining else t
                sleep_dur = next_arr - t
                self.timeline.append(('SLEEP', t, next_arr, 0, CPUState.SLEEP))
                self.total_energy += self.sleep_energy * sleep_dur
                t = next_arr
                continue

            # Pick next process (EDF-aware: choose closest deadline first among ready)
            ready_list = list(ready_q)
            # Urgency = remaining / slack
            def urgency(p):
                slack = p.deadline - t
                if slack <= 0:
                    return float('inf')
                return p.remaining / slack

            ready_list.sort(key=urgency, reverse=True)
            proc = ready_list[0]
            ready_q = deque([p for p in ready_q if p != proc])

            # Choose frequency via DVFS
            freq = DVFSModel.choose_freq(proc.remaining, proc.deadline, t)
            quantum = self._adaptive_quantum(proc)
            # How many real-time units this quantum takes at chosen freq
            real_units = min(proc.remaining, quantum)
            exec_t = math.ceil(DVFSModel.exec_time_at_freq(real_units, freq))

            if proc.start_time is None:
                proc.start_time = t

            # Execute
            start = t
            t += exec_t
            proc.remaining -= real_units
            energy = DVFSModel.energy(real_units, freq)
            proc.energy_consumed += energy
            self.total_energy += energy
            proc.frequency_used.append(freq)

            self.timeline.append((proc.pid, start, t, freq, CPUState.ACTIVE))

            if proc.remaining <= 0:
                proc.finish_time = t
                proc.turnaround_time = t - proc.arrival
                proc.waiting_time = proc.turnaround_time - proc.burst
            else:
                # Re-enqueue remaining
                ready_q.appendleft(proc)
                # Enqueue any newly arrived
                arrived2 = [p for p in remaining if p.arrival <= t]
                for p in arrived2:
                    ready_q.append(p)
                    remaining.remove(p)

        # Baseline energy (all at max freq, no DVFS)
        self.baseline_energy = sum(
            DVFSModel.energy(p.burst, 1.0) for p in self.processes
        )
        return self


# ─────────────────────────────────────────────
#  GUI Application
# ─────────────────────────────────────────────

FREQ_COLORS = {
    0.25: '#4CAF50',   # green  - very efficient
    0.50: '#8BC34A',   # light green
    0.75: '#FF9800',   # orange - moderate
    1.0:  '#F44336',   # red    - full power
}
SLEEP_COLOR = '#90CAF9'
IDLE_COLOR  = '#EEEEEE'

PROCESS_COLORS = [
    '#7E57C2','#42A5F5','#26A69A','#EC407A',
    '#FF7043','#66BB6A','#FFCA28','#AB47BC',
    '#26C6DA','#D4E157',
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Energy-Efficient CPU Scheduler  |  CSE-316  |  Krishan Kumar  |  Roll No: 35")
        self.geometry("1280x820")
        self.configure(bg='#1E1E2E')
        self.resizable(True, True)

        self.processes = []
        self.pid_counter = 1
        self.result = None

        self._build_ui()

    # ── UI Builder ────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self, bg='#1E1E2E')
        top.pack(fill='x', padx=12, pady=8)

        tk.Label(top, text="⚡ Energy-Efficient CPU Scheduler",
                 font=('Helvetica', 18, 'bold'), fg='#CDD6F4', bg='#1E1E2E').pack(side='left')
        tk.Label(top, text="Krishan Kumar  |  Roll No: 35  |  Reg: 12410760  |  R224EIB35  |  CSE-316",
                 font=('Helvetica', 10), fg='#A6ADC8', bg='#1E1E2E').pack(side='right')

        # ── Input Panel ──
        inp = tk.LabelFrame(self, text=" Add Process ", font=('Helvetica', 10, 'bold'),
                            fg='#CDD6F4', bg='#313244', bd=1, relief='groove')
        inp.pack(fill='x', padx=12, pady=4)

        fields = [
            ('Name',    'name_var',     'P1',   8),
            ('Arrival', 'arrival_var',  '0',    5),
            ('Burst',   'burst_var',    '5',    5),
            ('Priority (1-3)', 'prio_var', '2', 5),
            ('Deadline','deadline_var', '',     6),
        ]
        self.name_var     = tk.StringVar(value='P1')
        self.arrival_var  = tk.StringVar(value='0')
        self.burst_var    = tk.StringVar(value='5')
        self.prio_var     = tk.StringVar(value='2')
        self.deadline_var = tk.StringVar(value='')
        self.quantum_var  = tk.StringVar(value='4')
        self.power_var    = tk.StringVar(value='normal')

        for col, (label, var, default, w) in enumerate(fields):
            tk.Label(inp, text=label+':', fg='#CDD6F4', bg='#313244',
                     font=('Helvetica',9)).grid(row=0, column=col*2, padx=4, pady=6, sticky='e')
            tk.Entry(inp, textvariable=getattr(self, var), width=w,
                     bg='#45475A', fg='#CDD6F4', insertbackground='white').grid(
                row=0, column=col*2+1, padx=4)

        tk.Label(inp, text='Power Class:', fg='#CDD6F4', bg='#313244',
                 font=('Helvetica',9)).grid(row=0, column=10, padx=4)
        ttk.Combobox(inp, textvariable=self.power_var, values=['low','normal','high'],
                     width=7, state='readonly').grid(row=0, column=11, padx=4)

        tk.Label(inp, text='Quantum:', fg='#CDD6F4', bg='#313244',
                 font=('Helvetica',9)).grid(row=0, column=12, padx=4)
        tk.Entry(inp, textvariable=self.quantum_var, width=4,
                 bg='#45475A', fg='#CDD6F4', insertbackground='white').grid(row=0, column=13, padx=4)

        btn_f = tk.Frame(inp, bg='#313244')
        btn_f.grid(row=0, column=14, padx=8)
        tk.Button(btn_f, text='➕ Add', command=self._add_process,
                  bg='#89B4FA', fg='#1E1E2E', font=('Helvetica',9,'bold'),
                  relief='flat', padx=8).pack(side='left', padx=2)
        tk.Button(btn_f, text='🎲 Random', command=self._random_processes,
                  bg='#A6E3A1', fg='#1E1E2E', font=('Helvetica',9,'bold'),
                  relief='flat', padx=8).pack(side='left', padx=2)
        tk.Button(btn_f, text='🗑 Clear', command=self._clear,
                  bg='#F38BA8', fg='#1E1E2E', font=('Helvetica',9,'bold'),
                  relief='flat', padx=8).pack(side='left', padx=2)

        # ── Process Table ──
        tbl_frame = tk.Frame(self, bg='#1E1E2E')
        tbl_frame.pack(fill='x', padx=12, pady=4)
        cols = ('PID','Name','Arrival','Burst','Priority','Deadline','Power')
        self.tree = ttk.Treeview(tbl_frame, columns=cols, show='headings', height=4)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', background='#313244', foreground='#CDD6F4',
                        fieldbackground='#313244', rowheight=22)
        style.configure('Treeview.Heading', background='#45475A', foreground='#CDD6F4', font=('Helvetica',9,'bold'))
        widths = [40,80,60,60,70,70,70]
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor='center')
        self.tree.pack(side='left', fill='x', expand=True)
        sb = ttk.Scrollbar(tbl_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side='left', fill='y')
        self.tree.bind('<Delete>', lambda e: self._delete_selected())

        # ── Run Button ──
        run_f = tk.Frame(self, bg='#1E1E2E')
        run_f.pack(pady=4)
        tk.Button(run_f, text='▶  Run Energy-Efficient Scheduler', command=self._run,
                  bg='#CBA6F7', fg='#1E1E2E', font=('Helvetica',12,'bold'),
                  relief='flat', padx=20, pady=6).pack()

        # ── Notebook for results ──
        nb_frame = tk.Frame(self, bg='#1E1E2E')
        nb_frame.pack(fill='both', expand=True, padx=12, pady=4)
        style.configure('TNotebook', background='#1E1E2E', borderwidth=0)
        style.configure('TNotebook.Tab', background='#313244', foreground='#CDD6F4',
                        padding=[10,4], font=('Helvetica',9,'bold'))
        style.map('TNotebook.Tab', background=[('selected','#CBA6F7')],
                  foreground=[('selected','#1E1E2E')])
        self.nb = ttk.Notebook(nb_frame)
        self.nb.pack(fill='both', expand=True)

        self.gantt_frame  = tk.Frame(self.nb, bg='#1E1E2E')
        self.energy_frame = tk.Frame(self.nb, bg='#1E1E2E')
        self.stats_frame  = tk.Frame(self.nb, bg='#1E1E2E')
        self.nb.add(self.gantt_frame,  text=' 📊 Gantt Chart ')
        self.nb.add(self.energy_frame, text=' ⚡ Energy Analysis ')
        self.nb.add(self.stats_frame,  text=' 📋 Process Stats ')

    # ── Process Management ─────────────────────
    def _add_process(self):
        try:
            name     = self.name_var.get().strip() or f'P{self.pid_counter}'
            arrival  = int(self.arrival_var.get())
            burst    = int(self.burst_var.get())
            priority = int(self.prio_var.get())
            deadline = int(self.deadline_var.get()) if self.deadline_var.get() else None
            power    = self.power_var.get()
            assert 1 <= priority <= 3
            assert burst > 0
        except Exception:
            messagebox.showerror("Input Error", "Check your inputs.\nPriority: 1-3, Burst > 0.")
            return

        p = Process(self.pid_counter, name, arrival, burst, priority, deadline, power)
        self.processes.append(p)
        self.tree.insert('', 'end', iid=str(self.pid_counter),
                         values=(p.pid, p.name, p.arrival, p.burst,
                                 p.priority, p.deadline, p.power_class))
        self.pid_counter += 1
        # auto-increment name
        try:
            base = ''.join(filter(str.isalpha, name))
            num  = int(''.join(filter(str.isdigit, name)) or '0') + 1
            self.name_var.set(base + str(num))
        except Exception:
            pass

    def _random_processes(self):
        self._clear()
        n = random.randint(5, 8)
        t = 0
        for i in range(n):
            t += random.randint(0, 3)
            burst = random.randint(2, 10)
            prio  = random.randint(1, 3)
            dl    = t + burst + random.randint(burst, burst*3)
            pc    = random.choice(['low','normal','normal','high'])
            self.name_var.set(f'P{i+1}')
            self.arrival_var.set(str(t))
            self.burst_var.set(str(burst))
            self.prio_var.set(str(prio))
            self.deadline_var.set(str(dl))
            self.power_var.set(pc)
            self._add_process()

    def _delete_selected(self):
        sel = self.tree.selection()
        for iid in sel:
            pid = int(iid)
            self.processes = [p for p in self.processes if p.pid != pid]
            self.tree.delete(iid)

    def _clear(self):
        self.processes = []
        self.pid_counter = 1
        for row in self.tree.get_children():
            self.tree.delete(row)
        for f in (self.gantt_frame, self.energy_frame, self.stats_frame):
            for w in f.winfo_children():
                w.destroy()

    # ── Run Scheduler ─────────────────────────
    def _run(self):
        if not self.processes:
            messagebox.showwarning("No Processes", "Add at least one process first.")
            return
        try:
            q = int(self.quantum_var.get())
        except Exception:
            q = 4

        sched = EnergyEfficientScheduler(self.processes, base_quantum=q)
        sched.run()
        self.result = sched
        self._draw_gantt(sched)
        self._draw_energy(sched)
        self._draw_stats(sched)
        self.nb.select(0)

    # ── Gantt Chart ───────────────────────────
    def _draw_gantt(self, sched):
        for w in self.gantt_frame.winfo_children():
            w.destroy()

        pid_list = sorted(set(p.pid for p in sched.processes))
        pid_map  = {pid: i for i, pid in enumerate(pid_list)}
        proc_map = {p.pid: p for p in sched.processes}

        fig, ax = plt.subplots(figsize=(13, max(3, len(pid_list)*0.7 + 1.5)))
        fig.patch.set_facecolor('#1E1E2E')
        ax.set_facecolor('#313244')

        for entry in sched.timeline:
            pid, start, end, freq, state = entry
            if state == CPUState.SLEEP:
                ax.barh(0.5, end-start, left=start, height=0.6,
                        color=SLEEP_COLOR, alpha=0.5, edgecolor='none')
                ax.text((start+end)/2, 0.5, 'sleep', ha='center', va='center',
                        fontsize=7, color='#555')
                continue
            row = pid_map[pid] + 1
            color = PROCESS_COLORS[pid % len(PROCESS_COLORS)]
            bar = ax.barh(row, end-start, left=start, height=0.6,
                          color=color, alpha=0.85,
                          edgecolor=FREQ_COLORS[freq], linewidth=2)
            ax.text((start+end)/2, row, f'{proc_map[pid].name}\n{int(freq*100)}%',
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold')

        yticks = [0.5] + list(range(1, len(pid_list)+1))
        ylabels = ['CPU'] + [proc_map[pid].name for pid in pid_list]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, color='#CDD6F4')
        ax.set_xlabel('Time Units', color='#CDD6F4')
        ax.set_title('Gantt Chart  (border color = CPU frequency)', color='#CDD6F4', fontsize=11)
        ax.tick_params(colors='#CDD6F4')
        for spine in ax.spines.values():
            spine.set_edgecolor('#45475A')

        legend = [mpatches.Patch(color=FREQ_COLORS[f], label=f'{int(f*100)}% freq') for f in DVFSModel.LEVELS]
        legend.append(mpatches.Patch(color=SLEEP_COLOR, label='Sleep'))
        ax.legend(handles=legend, loc='lower right', fontsize=8,
                  facecolor='#45475A', labelcolor='#CDD6F4', framealpha=0.8)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.gantt_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)

    # ── Energy Chart ──────────────────────────
    def _draw_energy(self, sched):
        for w in self.energy_frame.winfo_children():
            w.destroy()

        procs = [p for p in sched.processes if p.finish_time is not None]
        if not procs:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        fig.patch.set_facecolor('#1E1E2E')
        for ax in (ax1, ax2):
            ax.set_facecolor('#313244')
            ax.tick_params(colors='#CDD6F4')
            for spine in ax.spines.values():
                spine.set_edgecolor('#45475A')

        # Bar: per-process energy vs baseline
        names    = [p.name for p in procs]
        actual   = [p.energy_consumed for p in procs]
        baseline = [DVFSModel.energy(p.burst, 1.0) for p in procs]
        x = range(len(names))
        ax1.bar([i-0.2 for i in x], baseline, 0.35, label='Baseline (full freq)',
                color='#F38BA8', alpha=0.8)
        ax1.bar([i+0.2 for i in x], actual,   0.35, label='DVFS actual',
                color='#A6E3A1', alpha=0.8)
        ax1.set_xticks(list(x)); ax1.set_xticklabels(names, color='#CDD6F4')
        ax1.set_ylabel('Energy (relative units)', color='#CDD6F4')
        ax1.set_title('Per-Process Energy: Baseline vs DVFS', color='#CDD6F4')
        ax1.legend(facecolor='#45475A', labelcolor='#CDD6F4')

        # Pie: freq distribution across all timeline entries
        freq_time = {f: 0 for f in DVFSModel.LEVELS}
        sleep_t   = 0
        for entry in sched.timeline:
            _, start, end, freq, state = entry
            if state == CPUState.SLEEP:
                sleep_t += end - start
            else:
                freq_time[freq] += end - start

        labels = [f'{int(f*100)}% freq' for f in DVFSModel.LEVELS] + ['Sleep']
        sizes  = [freq_time[f] for f in DVFSModel.LEVELS] + [sleep_t]
        colors = [FREQ_COLORS[f] for f in DVFSModel.LEVELS] + [SLEEP_COLOR]
        sizes_f = [s for s in sizes if s > 0]
        labels_f= [l for l, s in zip(labels, sizes) if s > 0]
        colors_f= [c for c, s in zip(colors, sizes) if s > 0]
        ax2.pie(sizes_f, labels=labels_f, colors=colors_f, autopct='%1.1f%%',
                textprops={'color':'#CDD6F4'}, startangle=90)
        savings = (1 - sched.total_energy/sched.baseline_energy)*100 if sched.baseline_energy else 0
        ax2.set_title(f'CPU Time Distribution\nEnergy Saved: {savings:.1f}%', color='#CDD6F4')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.energy_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)

    # ── Stats Table ───────────────────────────
    def _draw_stats(self, sched):
        for w in self.stats_frame.winfo_children():
            w.destroy()

        # Summary metrics
        procs = [p for p in sched.processes if p.finish_time is not None]
        avg_wait = sum(p.waiting_time for p in procs) / len(procs) if procs else 0
        avg_ta   = sum(p.turnaround_time for p in procs) / len(procs) if procs else 0
        savings  = (1 - sched.total_energy/sched.baseline_energy)*100 if sched.baseline_energy else 0

        summary = tk.Frame(self.stats_frame, bg='#313244')
        summary.pack(fill='x', padx=8, pady=6)
        metrics = [
            ('Avg Waiting Time',    f'{avg_wait:.2f} units'),
            ('Avg Turnaround Time', f'{avg_ta:.2f} units'),
            ('Total Energy (DVFS)', f'{sched.total_energy:.3f} units'),
            ('Baseline Energy',     f'{sched.baseline_energy:.3f} units'),
            ('Energy Saved',        f'{savings:.1f}%'),
        ]
        for i, (label, val) in enumerate(metrics):
            tk.Label(summary, text=label+':', fg='#A6ADC8', bg='#313244',
                     font=('Helvetica',10)).grid(row=0, column=i*2, padx=10, pady=6, sticky='e')
            tk.Label(summary, text=val, fg='#A6E3A1', bg='#313244',
                     font=('Helvetica',10,'bold')).grid(row=0, column=i*2+1, padx=4)

        # Per-process table
        cols = ('PID','Name','Arrival','Burst','Finish','Wait','Turnaround','Energy','Avg Freq')
        style = ttk.Style()
        style.configure('Stats.Treeview', background='#313244', foreground='#CDD6F4',
                        fieldbackground='#313244', rowheight=24)
        style.configure('Stats.Treeview.Heading', background='#45475A',
                        foreground='#CDD6F4', font=('Helvetica',9,'bold'))
        tree = ttk.Treeview(self.stats_frame, columns=cols, show='headings',
                            height=12, style='Stats.Treeview')
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=90, anchor='center')
        for p in sched.processes:
            avg_f = sum(p.frequency_used)/len(p.frequency_used) if p.frequency_used else 0
            tree.insert('', 'end', values=(
                p.pid, p.name, p.arrival, p.burst,
                p.finish_time if p.finish_time else 'N/A',
                max(0, p.waiting_time), p.turnaround_time,
                f'{p.energy_consumed:.3f}', f'{avg_f:.2f}'
            ))
        tree.pack(fill='both', expand=True, padx=8, pady=4)


# ─────────────────────────────────────────────
if __name__ == '__main__':
    app = App()
    app.mainloop()
