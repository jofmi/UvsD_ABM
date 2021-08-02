import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from UvsD_ABM_Model import calc_local_abatement_analysis


fs = 13 # Font size for plots

custom_lines = [Line2D([0], [0], color="C0", lw=2),
                Line2D([0], [0], color="C1", lw=2, ls="--"),
                Line2D([0], [0], color="C2", lw=2, ls="--"),
                Line2D([0], [0], color="C3", lw=2, ls="--"),
                Line2D([0], [0], color="C1", lw=2),
                Line2D([0], [0], color="C2", lw=2),
                Line2D([0], [0], color="C3", lw=2),]

target_line = [Line2D([0], [0], color="grey", ls=":", lw=2)]

custom_labels = ["No Policy","Upstream Tax","Upstream Permit market",
                 "Upstream Direct Regulation","Downstream Tax",
                 "Downstream Permit market","Downstream Direct Regulation"]
    
    
def plot_abatement_analysis(results, domestic=False):

    sec,p,usec=results[0]

    t0= p.t_start-1 
    T = p.T 
    ts = range (T-t0)
    
    fig, ([[ax0,ax2,ax1],[ax3,ax5,ax4],[ax7,ax8,ax9]]) = plt.subplots(nrows=3, ncols=3, figsize=(11, 7.5))
    axs = [ax0,ax1,ax2,ax3,ax4,ax5,ax6]
    
    label = "Abatement ($α$)"
    if domestic: label = "Local " + label
        
    ax0.set_ylabel(label, fontsize = fs) 
    ax3.set_ylabel(label, fontsize = fs) 
    
    i = 0 
    for sc in results:
        
        sec,p,usec=sc
        
        axs[i].set_title(p.label, fontsize = fs)

        if domestic: ab_21, ab_1,ab_22,ab_tot = calc_local_abatement_analysis(sc,t0)
        else: ab_21, ab_1,ab_22,ab_tot = calc_abatement_analysis(sc,t0)

        pal = ["#bc5090","#ffa600", "#58508d", "#003f5c"] 
        stacks = axs[i].stackplot(ts, ab_21, ab_1,ab_22,  labels=["Compositional change","Technology adoption","Reduction of production"], alpha=0.8, colors=pal) 
        axs[i].axhline(sec.E_cov[t0]-p.reg.E_max,color='grey',ls='--',label='Abatement Target') # Abatement Target Line
        axs[i].plot(ts,ab_tot,label="Total Abatement")
        axs[i].set_xlabel('Time (t)', fontsize = fs)
        axs[i].grid()
        axs[i].set_xlim([0,T-t0])
        axs[i].set_ylim([0,0.7])

        i+=1

    ax0.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('Outputs/abatement.pdf')
    
    
def fplot(variable,label,mode,ax,sec,p):
    
    start = 1
    end = p.T
    ts = range(start,end) 

    for i in range(len(sec)):
        exec("ax.plot(ts,[sec["+str(i)+"]."+variable+"[t] for t in ts], label=f'Producer {sec["+str(i)+"].j}')") 

    ax.set_xlabel('time (t)')
    ax.set_ylabel(label)
    ax.set_xticks(np.arange(start,end, 1),minor = True)
    ax.grid()
