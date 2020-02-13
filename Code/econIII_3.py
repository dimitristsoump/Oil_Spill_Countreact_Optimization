'''=================================================================================================
Required Libraries
================================================================================================='''
import numpy as np
import matplotlib.pyplot as plt
#from staticmap import StaticMap, CircleMarker
'''=================================================================================================
Base classes
================================================================================================='''
# Prestige oil outflow happened 130 miles from the shore of Galicia, so only booms and skimmers are gonna be used
class skimmers:
    def __init__(self,vol_inflow,capacity,num):
        self.vi=vol_inflow #m^3/h
        self.cap=capacity #m^3
        self.num=num # how many we have
        self.tnum=num
        self.used=[]
    def use(self,howmany):
        if (self.num!=0):
            self.num-=howmany
    def replenish(self):
        self.num=self.tnum
        #self.used=[]

class booms:
    def __init__(self,tens_strength,length,freeboard):
        self.tenstrength=tens_strength #kg
        self.len=length #m
        self.freeboard=freeboard #cm
        self.num=np.random.randint(0,600) # how many we have
        self.radius=[]
    def use(self):
        if (self.num!=0):
            self.num-=1
'''================================================================================================
Needed Definitions
================================================================================================'''
distance_from_closest_coast=250 #kmr
tug_speed=14*0.5144 #14 kn
# Adios results
xremain=np.linspace(0, 72, num=12)
yremain=np.array([100,77,71,70,69,68,67,65,63,62,61,60])*59305/100
xdisp=np.linspace(0, 72, num=12)
ydisp=np.array([0,5.5,9.5,11,11.2,11.3,11.4,11.5,11.6,11.7,11.9,12])*59305/100
xremain3=np.linspace(0, 72, num=24)
xdisp3=np.linspace(0, 72, num=24)
yremain=np.polyval(np.polyfit(xremain,yremain,2),xremain3)
ydisp=np.polyval(np.polyfit(xdisp,ydisp,2),xdisp3)
#remaining_curve=np.polyfit(xremain3,yremain,2) #2nd degree polyonymials are chosen for the curves
#disperse_curve=np.polyfit(xdisp3,ydisp,2)

#Goal for the first 72 hours
xremain_std=np.linspace(0, 72, num=12)
yremain_std=np.array([100,70,65,63,60,59,57,56,55,54,52,50])*59305/100
#yremain_std=np.array([100,60,57,48,45,40,38,33,30,28,26,24])*59305/100
xdisp_std=np.linspace(0, 72, num=12)
ydisp_std=np.array([0,1,1.1,1.2,1.25,1.3,1.4,1.45,1.5,1.7,1.9,2])*59305/100
xremain3_std=np.linspace(0, 72, num=24)
xdisp3_std=np.linspace(0, 72, num=24)
yremain_std=np.polyval(np.polyfit(xremain_std,yremain_std,2),xremain3_std)
ydisp_std=np.polyval(np.polyfit(xdisp_std,ydisp_std,2),xdisp3_std)
#remaining_curve_std=np.polyfit(xremain_std,yremain_std,2) #2nd degree polyonymials are chosen for the curves
#disperse_curve_std=np.polyfit(xdisp_std,ydisp_std,2)

#Plotting
plt.figure()
plt.plot(xremain3,yremain,'b',label="Real")
plt.plot(xremain3_std,yremain_std,'g',label="Goal")
plt.xlabel("Time passed [h]")
plt.ylabel("Oil Volume [metric tons]")
plt.title("Remaining Oil Volume Curve")
plt.grid()
plt.legend(loc="upper right")
plt.show()
plt.figure()
plt.plot(xdisp3,ydisp,'b',label="Real")
plt.plot(xdisp3_std,ydisp_std,'g',label="Goal")
plt.xlabel("Time passed [h]")
plt.ylabel("Oil Volume [metric tons]")
plt.title("Dispersed Oil Volume Curve")
plt.grid()
plt.legend(loc="upper right")
plt.show()
'''================================================================================================
Helper functions
================================================================================================'''
def create_individuals(howmany,fromold):
    individuals=[]
    if (not fromold):
        d=howmany
    else:
        d=howmany-len(fromold)
    for ii in range(0,d):
        skim_small=skimmers(15,100,np.random.randint(1,100)) #https://www.globalspill.com.au/product/weir-skimmer-30000-lhr-gsw30esp/
        skim_large=skimmers(30,100,np.random.randint(1,100))
        boom=booms(16329,1,25.4) #https://www.abasco.com/boomsigma.html Sigma 24
        shore_boom=booms(16329,1,35.6) # https://www.abasco.com/boomshoreline.html
        individuals.append([skim_small,skim_large,boom])
    if fromold:
        for i in fromold:
            individuals.append(i)
    return individuals

def surf_thick(m,t): # Den mporoume na ypothesoume tis A5th, A50th gia thour=0 opote t>=1
    vol=m/0.863
    thour=np.array([1,2,5,10,24,48,72]) #hours
    A05th=np.array([0.076,0.107,0.169,0.24,0.68,1.93,3.54,5.45,64.8]) #km^2
    A5th=np.array([0.36,0.496,0.784,1.11,1.72,2.43,3.54]) #km^2
    A50th=np.array([1.14,2.28,3.64,5.15,7.98,11.3,13.8])  # km^2
    thick05=np.array([7.5,5.3,3.4,2.4,0.84,0.3,0.16,0.105,0.009]) # mm
    thick5=np.array([15.8,11.5,7,5.1,3.3,2.4,1.6]) # mm
    thick50=np.array([50.1,25.1,15.7,11.1,7.2,5.1,4.1]) # mm
    vol1=5000/0.863
    vol2=10*vol1
    vol3=vol1/10
    i=0
    for k in range(len(thour)):
        if t<=thour[k] :
            break
        else:
            i+=1
            if i==len(thour):
                i-=1
    A05=A05th[i-1]+(A05th[i]-A05th[i-1])/(thour[i]-thour[i-1])*(t-thour[i-1])
    A5=A5th[i-1]+(A5th[i]-A5th[i-1])/(thour[i]-thour[i-1])*(t-thour[i-1])
    A50=A50th[i-1]+(A50th[i]-A50th[i-1])/(thour[i]-thour[i-1])*(t-thour[i-1])
    thick05=thick05[i-1]+(thick05[i]-thick05[i-1])/(thour[i]-thour[i-1])*(t-thour[i-1])
    thick5=thick5[i-1]+(thick5[i]-thick5[i-1])/(thour[i]-thour[i-1])*(t-thour[i-1])
    thick50=thick50[i-1]+(thick50[i]-thick50[i-1])/(thour[i]-thour[i-1])*(t-thour[i-1])
    if (m>=5000):
        Avol=A5+(A50-A5)/(vol2-vol1)*(vol-vol1)
        tvol=thick5+(thick50-thick5)/(vol2-vol1)*(vol-vol1)
    elif ((m<5000) and (m>500)):
        Avol=A05+(A5-A05)/(vol1-vol3)*(vol-vol3)
        tvol=thick05+(thick5-thick05)/(vol1-vol3)*(vol-vol3)
    else:
        print ("Out of bounds")
        Avol=A5+(A50-A5)/(vol2-vol1)*(vol-vol1)
        tvol=thick5+(thick50-thick5)/(vol2-vol1)*(vol-vol1)

    return Avol,tvol

def oil_rec_rate(max_rate,sl_thick):
    ratio=max_rate/20.16 #to normalize the capacity (20.16 is the maximu capacity of an experiment)
    thick=np.array([0,6,12,25,50,62.5])
    cap=np.array([0,59,87,170,272,336])*(60/1000)
    rec_eff=np.array([0,75,80,81,87,82])
    i=0
    for k in range(len(thick)):
        if sl_thick<=thick[k] :
            break
        else:
            i+=1
    capacity=cap[i-1]+(cap[i]-cap[i-1])/(thick[i]-thick[i-1])*(sl_thick-thick[i-1])
    capacity=capacity*ratio
    rec_efficiency=rec_eff[i-1]+(rec_eff[i]-rec_eff[i-1])/(thick[i]-thick[i-1])*(sl_thick-thick[i-1])
    return [capacity,rec_efficiency]

def cross_over(couples):
    # 2 Individuals from gen>1 are produced by cross over from 2 couples
    couple1=[couples[0][1],couples[1][0]]
    couple2=[couples[0][0],couples[1][1]]
    children=[]
    for couple in [couple1,couple2]:
        father=couple[0]
        mother=couple[1]
        skim_small=skimmers(15,100,int(father[0].tnum*0.8+mother[0].tnum*0.2)) #https://www.globalspill.com.au/product/weir-skimmer-30000-lhr-gsw30esp/
        skim_large=skimmers(30,100,int(father[0].tnum*0.8+mother[0].tnum*0.2))
        boom=booms(16329,1,25.4) #https://www.abasco.com/boomsigma.html Sigma 24
        children.append([skim_small,skim_large,boom])
    return children


def mutation(ind):
    # Here we change the number of the equipment by a certain probability px 0.4
    mutation_gain=1
    base_prob_min=0.1
    base_prob_max=0.8
    for item in ind:
        prob=np.random.rand()
        if (prob<base_prob_min):
            item[0].num-=mutation_gain
            item[1].num+=mutation_gain
        elif (prob>base_prob_max):
            item[0].num+=mutation_gain
            item[1].num-=mutation_gain
        else:
            pass
    return ind

def choose_for_cross(generation,fitness):
    # 2 Couples are gonna be formed in each generation
    chosen_fathers=0
    chosen_mothers=0
    gen_sorted = sorted(generation, key=lambda x: fitness)
    fitness_sorted=sorted(fitness)
    fathers=[generation[fitness.index(min(fitness))],gen_sorted[1]]
    mothers=[]
    for jj in range (0,2):
        index_s=np.random.randint(2,len(gen_sorted), size=1)
        index_s=index_s[0]
        mothers.append(gen_sorted[index_s])
    couples=[fathers,mothers]
    return couples


def compute_fitness(generation):
    # Fitness is a metric that allows us to judge the different individuals and compare their results
    fit_table=[]
    length_table=[]
    result_rems=[]
    nss=[]
    nls=[]
    for unit in generation:
        metric1,metric2,boom_length,result_rem,ns,nl=strategy(unit)
        fitness=metric1*boom_length
        fit_table.append(fitness)
        length_table.append(boom_length)
        result_rems.append(result_rem)
        nss.append(ns)
        nls.append(nl)
    rem_to_draw=result_rems[fit_table.index(min(fit_table))]
    ns=nss[fit_table.index(min(fit_table))]
    nl=nls[fit_table.index(min(fit_table))]
    return fit_table,length_table,rem_to_draw,ns,nl

def add_reuse(int_supp,row_reu,row_loa,demm,types):#genika prepei len(row_reu)==n_reu
    n_reu=len(row_reu)
    n_loa=len(row_loa)
    a=row_loa.pop()
    int_supp+=a
    if (types=='s'):
        if (sum(row_reu)<demm):
            demm-=sum(row_reu)
            if (int_supp>=demm):
                int_supp-=demm
                row_reu.insert(0,demm)
                b=row_reu.pop()
                row_loa.insert(0,b)
                demm=0
                
            else:
                demm-=int_supp
                row_reu.insert(0,int_supp)
                b=row_reu.pop()
                row_loa.insert(0,b)
                int_supp=0
            
        else:
            for k in range(len(row_reu)-1,-1,-1): #psaxnoume apo pisw pros ta mprws gia to prwto meriko athroisma pou tha vgalei
                                                      #thn demmanding posothta
                if (sum(row_reu[k:len(row_reu)])>=demm):
                    if (k==len(row_reu)-1):
                        row_reu[k]-=demm
                        row_loa.insert(0,demm)
                        break
                    elif (k<len(row_reu)-1):
                        a=demm-sum(row_reu[k+1:len(row_reu)])
                        #print(a,k)
                        row_reu[k]-=a
                        row_reu.insert(k+1,a)
                        #print(row_reu)
                        b=row_reu.pop()
                        row_loa.insert(0,b)
                        break
    else:
        if (demm>int_supp):
            row_loa.insert(0,int_supp)
            int_supp=0
        else:
            row_loa.insert(0,demm)
            int_supp-=demm
    return int_supp,row_reu,row_loa

def strategy(unit):
    '''
    -Input: unit [random distribution of equipment]
    -Output: rem_control_metric [how close we got to a specified standard for the oil remaining at sea],
            disp_control_metric [how close we got to a specified standard for the oil limitation]
    -Means: losses curve and limitation curve. This curves are going to be substracted from the remaining and disperse curves respectively
    every single hour. As a result, 2 new curves will be produced and compared with the standard remaining and the standard disperse curves.
    *rem_control_metric= mean(std_deviations between the new remaining curve and the standard one for each hour)
    *disp_control_metric= mean(std_deviations between the new disperse curve and the standard one for each hour)
    - Hypotheses: 1/linear absorption from skimmers
                  2/real equipment use rules
    '''
    skimm_s=unit[0]
    skimm_l=unit[1]
    ach_rem=[]
    time_grid=np.linspace(0,72,24)#Render grid to compute losses and limitation curves
    diff_rem=np.subtract(yremain,yremain_std)
    diff_disp=np.subtract(ydisp,ydisp_std)
    rem_req=[]
    disp_req=[]
    time_small=skimm_s.cap/skimm_s.vi # how many hours it takes
    time_large=skimm_l.cap/skimm_l.vi # how many hours it takes
    time_for_reuse=2*distance_from_closest_coast/(tug_speed*3600/1000)
    time_step=3 # the time steps we choose
    num_of_time_steps=int(np.ceil(time_for_reuse/time_step))
    t_steps_skimm_s=round(time_small/time_step) # how many time increments we are capable of using small skimmers
    t_steps_skimm_l=round(time_large/time_step)# how many time increments we are capable of using large skimmers
    row_skimm_s=np.zeros(num_of_time_steps).tolist() # number of skimmers_s that will be available for reuse
    row_reuse_s=np.zeros(t_steps_skimm_s).tolist() # number of skimmers that can still be reused
    row_skimm_l=np.zeros(num_of_time_steps).tolist() # number of skimmers_l that will be available for reuse
    row_reuse_l=np.zeros(t_steps_skimm_l).tolist() # number of skimmers that can still be reused
    for ii in range (0,len(time_grid)-1):
        if (diff_rem[ii]>=diff_rem[ii+1]):
            rem_req_ii=diff_rem[ii]
        else:
            rem_req_ii=diff_rem[ii]*0.3+diff_rem[ii+1]*0.7 # Future goal accounts more than the current one
        disp_req.append(diff_disp[ii])
        rem_req.append(rem_req_ii)
    rem_req.append(diff_rem[len(diff_rem)-1])
    supplies_s=[]
    supplies_l=[]
    # Equipment use for remaining goal
    for goal in rem_req:
        large=0
        small=0
        # oil density 0.863 metric tons/m^3 70% of the needs will be covered by large skimmers and 30% by small ones
        # No consideration for capacity!!!!! Must be fixed
        remain_small=0
        remain_large=0
        num_large=np.ceil(0.7*goal/(skimm_l.vi*0.863*3))
        num_small=np.ceil(0.3*goal/(skimm_s.vi*0.863*3))
        if (num_large>skimm_l.num):
            remain_large=(num_large-skimm_l.num)*skimm_l.vi*0.863*3
            num_large=skimm_l.num
        #skimm_l.use(num_large)
        if (num_small>skimm_s.num):
            remain_small=(num_small-skimm_s.num)*skimm_s.vi*0.863*3
            num_small=skimm_s.num
        #skimm_s.use(num_small)
        large=num_large # first estimation of tottal number o skimm_l
        small=num_small # first estimation of tottal number o skimm_s
        if (skimm_l.num==0): # trying to cover the skimm_s demand by skimm_l supply
            pass
        else:
            if (remain_small<=skimm_l.vi*0.863*3*skimm_l.num):
                num_large=np.ceil(remain_small/(skimm_l.vi*0.863*3))
                #skimm_l.use(num_large)
                large+=num_large
            else:
                large+=skimm_l.num # the tottal number of skimm_l being used in this time step
                #skimm_l.use(skimm_l.num)
        if (skimm_s.num==0): # trying to cover the skimm_l demand by skimm_s supply
            pass
        else:
            if (remain_large<=skimm_s.vi*0.863*3*skimm_s.num):
                num_small=np.ceil(remain_large/(skimm_s.vi*0.863*3))
                #skimm_s.use(num_small)
                small+=num_small
            else:
                small+=skimm_s.num # the tottal number of skimm_s being used in this time step
                #skimm_s.use(skimm_s.num)
        [num_small_rem,row_reuse_s,row_skimm_s]=add_reuse(skimm_s.num,row_reuse_s,row_skimm_s,small,'s')
        [num_large_rem,row_reuse_l,row_skimm_l]=add_reuse(skimm_l.num,row_reuse_l,row_skimm_l,large,'l')
        supplies_s.append(num_small_rem)
        supplies_l.append(num_large_rem)
        skimm_s.num=num_small_rem
        skimm_l.num=num_large_rem
        ach_rem.append((skimm_s.tnum-skimm_s.num)*skimm_s.vi*0.863*3+ (skimm_l.tnum-skimm_l.num)*skimm_l.vi*0.863*3)
        skimm_s.used.append(small)
        skimm_l.used.append(large)
    needed_small=skimm_s.tnum-min(supplies_s)
    needed_large=skimm_l.tnum-min(supplies_l)
    ach_rem=np.array(ach_rem)
    result_rem=np.subtract(yremain,ach_rem)
    diff_rem_from_std=np.abs(np.subtract(result_rem,yremain_std))
    rem_control_metric=np.mean(diff_rem_from_std)
    for jj in range (0,len(time_grid)):
        t=time_grid[jj]
        indexes=[x for x in range (0,jj)]
        sum_ach=ach_rem[indexes].sum()
        surface,thickness=surf_thick(59305-sum_ach,t)
        radius=np.sqrt(surface/np.pi)
        unit[2].radius.append(radius)
        perimeter=2*np.pi*radius
        total_distance_for_tug=distance_from_closest_coast+perimeter/2
        time_to_cover=(total_distance_for_tug*1000/tug_speed)/3600
        if (time_to_cover<=t):
            break
    disp_control_metric=time_to_cover
    return rem_control_metric, disp_control_metric,perimeter,result_rem,needed_small,needed_large

def render_map(ind,blength):
    wreck=[160.2/4,169/4]
    storage=[308/4,171.4/4]
    scale=np.sqrt((storage[1]**2-wreck[1]**2)+(storage[0]**2-wreck[0]**2))/distance_from_closest_coast
    img = plt.imread("map.png")
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 100, 0, 75])
    w,=ax.plot(wreck[0],wreck[1],'rx')
    st,=ax.plot(storage[0],storage[1],'rx')
    ax.plot([wreck[0],storage[0]],([wreck[1],storage[1]]),'b--')
    skimm_s=ind[0]
    skimm_l=ind[1]
    booms=ind[2]
    theta = np.linspace(0, 2*np.pi, 100).tolist()
    radius=[]
    for time in xremain3[0:len(xremain3):4]:
        A,_=surf_thick(yremain[xremain3.tolist().index(time)],time)
        r=np.sqrt(A/np.pi)
        radius.append(r)
    arc_length=30*np.pi/180
    arc_spoint=15*np.pi/180
    spoints=[]
    fpoints=[]
    for R in radius:
        xc=[wreck[0]+R*scale*np.cos(th) for th in theta]
        yc=[wreck[1]+R*scale*np.sin(th) for th in theta]
        ax.plot(xc,yc)
        index=radius.index(R)
        s=skimm_s.used[4*index]+skimm_s.used[4*index+1]+skimm_s.used[4*index+2]+skimm_s.used[4*index+3]
        l=skimm_l.used[4*index]+skimm_l.used[4*index+1]+skimm_l.used[4*index+2]+skimm_l.used[4*index+3]
        fi_s=np.linspace(0,2*np.pi,s,endpoint=False).tolist()
        fi_l=np.linspace(0,2*np.pi,l,endpoint=False).tolist()
        if index==0:
            r=R/2
        else:
            r=(radius[index]+radius[index-1])/2
        xps=[wreck[0]+3*r*scale/4*np.cos(fi) for fi in fi_s]
        yps=[wreck[1]+3*r*scale/4*np.sin(fi) for fi in fi_s]

        xpl=[wreck[0]+r*scale/4*np.cos(fi) for fi in fi_l]
        ypl=[wreck[1]+r*scale/4*np.sin(fi) for fi in fi_l]
        small_plot,=ax.plot(xps,yps,'g^')
        large_plot,=ax.plot(xpl,ypl,'rs')
        #Boom placement
        arc_thetas=[]
        if (index==0):
            spoints.append(arc_spoint)
            fpoints.append(arc_spoint+arc_length)
            arc_thetas=[np.linspace(spoints[0],fpoints[0],100)]
        else:
            for s in spoints:
                arc_theta=np.linspace(s-arc_length,s,100)
                arc_thetas.append(arc_theta)
            for f in fpoints:
                arc_theta=np.linspace(f,f+arc_length,100)
                arc_thetas.append(arc_theta)
            spoints=[arc_thetas[i][0] for i in range(0,len(arc_thetas))]
            fpoints=[arc_thetas[i][-1] for i in range(0,len(arc_thetas))]
        for i in range (0,len(arc_thetas)):
            arc_x=[wreck[0]+R*scale*np.cos(th) for th in arc_thetas[i]]
            arc_y=[wreck[1]+R*scale*np.sin(th) for th in arc_thetas[i]]
            booms,=ax.plot(arc_x,arc_y,'k',linewidth=2)
    ax.legend((w,st,small_plot,large_plot,booms),('Wreck','Storage','Small skimmers','Large skimmers','Boom'))
    plt.show()
'''================================================================================================
Main function
================================================================================================'''
def main():
    # Genetic Algorithm
    n=20 # We are gonna have 9 individuals in each gen
    maxgens=200
    bestfitness_old=0
    old=[]
    plt.ion()
    for gen in range(0,maxgens):
        ind=create_individuals(n,old)
        fitness,boom_length,rem,ns,nl=compute_fitness(ind)
        min_fit=min(fitness)
        #print(fitness)
        print("Generation "+str(gen+1)+"/"+ str(maxgens)+" Fitness: " +str(round(min_fit,4)))
        plt.clf()
        plt.plot(xremain3,rem,'r',label="Achieved")
        plt.plot(xremain3,yremain,'b',label="No intervention")
        plt.plot(xremain3,yremain_std,'g',label="Goal")
        plt.legend()
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("W [tons]")
        plt.title("Achieved results for gen "+str(gen+1))
        plt.show()
        plt.pause(0.0001)
        if (np.abs(min_fit-bestfitness_old)<=10**(-6)):
            break
        else:
            bestfitness_old=min_fit
        for ii in range (0,len(ind)):
            for jj in range (0,2):
                ind[ii][jj].replenish()
        couples=choose_for_cross(ind,fitness)
        new_ind=cross_over(couples)
        new_ind=mutation(new_ind)
        new_ind=[]
        ind[fitness.index(min(fitness))][0].replenish()
        ind[fitness.index(min(fitness))][1].replenish()
        old=new_ind.append(ind[fitness.index(min(fitness))]) #The best individual from the previous gen remains
    plt.ioff()
    print ("Results")
    print("========")
    fittest_ind=ind[fitness.index(min(fitness))]
    #no_shore_booms=fittest_ind[3].num
    boom_l=boom_length[fitness.index(min(fitness))]
    print(fittest_ind[1].used)
    no_large_skimmers=fittest_ind[1].tnum
    no_small_skimmers=fittest_ind[0].tnum
    print ("Maximum fitness achieved: "+ str(round(min(fitness),4)))
    print ("Number of small skimmers needed: "+ str(ns))
    print ("Number of large skimmers needed: "+ str(nl))
    print ("Length of booms needed: "+ str(round(boom_l,2))+" km")
    #print ("Number of shore booms needed: "+ str(no_shore_booms))
    #plt.figure() #Figure showing our actions for the 3 first days
    render_map(fittest_ind,boom_l) # Rendering of equipment placing on a real map


# Keeping things organised
if __name__=='__main__':
    main()
