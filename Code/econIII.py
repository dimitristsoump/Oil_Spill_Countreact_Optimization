'''=================================================================================================
Required Libraries
================================================================================================='''
import numpy as np
import matplotlib.pyplot as plt
from staticmap import StaticMap, CircleMarker
'''=================================================================================================
Base classes
================================================================================================='''
# Prestige oil outflow happened 130 miles from the shore of Galicia, so only booms and skimmers are gonna be used
class skimmers:
    def __init__(self,vol_inflow,capacity):
        self.vi=vol_inflow #m^3/h
        self.cap=capacity #m^3
        self.num=np.random.randint(0,300) # how many we have
    def use(self,howmany):
        if (self.num!=0):
            self.num-=howmany

class booms:
    def __init__(self,tens_strength,length,freeboard):
        self.tenstrength=tens_strength #kg
        self.len=length #m
        self.freeboard=freeboard #cm
        self.num=np.random.randint(0,600) # how many we have
    def use(self):
        if (self.num!=0):
            self.num-=1
'''================================================================================================
Needed Definitions
================================================================================================'''
# Adios results
xremain=np.linspace(0, 72, num=12)
yremain=np.array([100,77,71,70,69,68,67,65,63,62,61,60])*59305/100
xdisp=np.linspace(0, 72, num=12)
ydisp=np.array([0,5.5,9.5,11,11.2,11.3,11.4,11.5,11.6,11.7,11.9,12])*59305/100
remaining_curve=np.polyfit(xremain,yremain,2) #2nd degree polyonymials are chosen for the curves
disperse_curve=np.polyfit(xdisp,ydisp,2)

#Goal for the first 72 hours
xremain_std=np.linspace(0, 72, num=12)
yremain_std=np.array([100,72,68,53,45,40,34,30,27,25,23,17])*59305/100
xdisp_std=np.linspace(0, 72, num=12)
ydisp_std=np.array([0,1,1.1,1.2,1.25,1.3,1.4,1.45,1.5,1.7,1.9,2])*59305/100
remaining_curve_std=np.polyfit(xremain_std,yremain_std,2) #2nd degree polyonymials are chosen for the curves
disperse_curve_std=np.polyfit(xdisp_std,ydisp_std,2)

#Plotting
plt.figure()
plt.plot(xremain,yremain,'b',label="Real")
plt.plot(xremain_std,yremain_std,'g',label="Goal")
plt.xlabel("Time passed [h]")
plt.ylabel("Oil Volume [metric tons]")
plt.title("Remaining Oil Volume Curve")
plt.grid()
plt.legend(loc="upper right")
plt.show()
plt.figure()
plt.plot(xdisp,ydisp,'b',label="Real")
plt.plot(xdisp_std,ydisp_std,'g',label="Goal")
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
    for ii in range(0,howmany-len(fromold)):
        skim_small=skimmers(15,100) #https://www.globalspill.com.au/product/weir-skimmer-30000-lhr-gsw30esp/
        skim_large=skimmers(30,100)
        boom=booms(16329,1,25.4) #https://www.abasco.com/boomsigma.html Sigma 24
        shore_boom=booms(16329,1,35.6) # https://www.abasco.com/boomshoreline.html
        individuals.append([skim_small,skim_large,boom,shore_boom])
    if fromold:
        for i in fromold:
            individuals.append(i)
    return individuals

def cross_over():
    # 2 Individuals from gen>1 are produced by cross over from 2 couples
    pass

def mutation():
    # Here we change the number of the equipment by a certain probability px 0.4
    pass

def choose_for_cross(generation):
    # 2 Couples are gonna be formed in each generation
    pass

def compute_fitness(generation):
    # Fitness is a metric that allows us to judge the different individuals and compare their results
    fit_table=[]
    for unit in generation:
        metric1,metric2=strategy(unit)
        fitness=metric1*metric2
        fit_table.append(fitness)
    return fit_table

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
    time_grid=np.linspace(0,72,12)#Render grid to compute losses and limitation curves
    diff_rem=np.subtract(yremain,yremain_std)
    diff_disp=np.subtract(ydisp,ydisp_std)
    rem_req=[]
    disp_req=[]
    for ii in range (0,time_grid.size()-1):
        rem_req_ii=diff[ii]*0.3+diff[ii+1]*0.7 # Future goal accounts more than the current one
        disp_req.append(diff_disp[ii])
        rem_req.append(rem_req_ii)
    rem_req.append(diff_rem[diff_rem.size()])
    # Equipment use for remaining goal
    skimm_s=unit[0]
    skimm_l=unit[1]
    ach_rem=[]
    for goal in rem_req:
        # oil density 0.863 metric tons/m^3 70% of the needs will be covered by large skimmers and 30% by small ones
        # No consideration for capacity!!!!! Must be fixed
        num_large=np.floor(0.7*goal/(skimm_l.vi*0.863*12))
        num_small=np.floor(0.3*goal/(skimm_s.vi*0.863*12))
        if (num_large>skimm_l.num):
            num_large=skimm_l.num
            skimm_l.use(num_large)
        if (num_small>skimm_s.num):
            num_small=skimm_s.num
            skimm_s.use(num_small)
        time_small=skimm_s.cap/skimm_s.vi # how many hours it takes
        time_large=skimm_l.cap/skimm_l.vi # how many hours it takes
        ach_rem.append(num_small*skimm_s.vi*0.863*12+ num_large*skimm_l.vi*0.863*12)
    ach_rem=np.array(ach_rem)
    result_rem=np.subtract(yremain,ach_rem)
    diff_rem_from_std=np.subtract(result_rem,yremain_std)
    rem_control_metric=np.mean(diff_rem_from_std)


    disp_control_metric=None
    return rem_control_metric, disp_control_metric

def render_map():
    m = StaticMap(400, 400, url_template='http://a.tile.osm.org/{z}/{x}/{y}.png')

    marker_outline = CircleMarker((42.883333, -9.883333), 'white', 18)
    marker = CircleMarker((42.883333, -9.883333), '#0036FF', 12)

    m.add_marker(marker_outline)
    m.add_marker(marker)
    image = m.render()
    image.save('map.png')

'''================================================================================================
Main function
================================================================================================'''
def main():
    # Genetic Algorithm
    render_map() # Rendering of equipment placing on a real map
    n=9 # We are gonna have 9 individuals in each gen
    maxgens=10000
    bestfitness_old=0
    old=[]
    for gen in range(0,maxgens):
        ind=create_individuals(n,old)
        fitness=compute_fitness(ind)
        max_fit=max(fitness)
        print("Generation "+str(gen+1)+"/"+ str(maxgens)+" Fitness: " +str(max_fit))
        if (np.abs(max_fit-bestfitness_old)<=10^(-5)):
            print(" A solution that meets the criteria was found. Exiting algorithm...")
            break
        else:
            bestfitness_old=max_fit
        couples=choose_for_cross(ind,fitness)
        new_ind=cross_over(couples)
        new_ind=mutation(new_ind)
        old=new_ind.append(ind[fitness.index(max(fitness))]) #The best individual from the previous gen remains
    print ("Results")
    print("========")
    fittest_ind=ind[fitness.index(max(fitness))]
    no_shore_booms=fittest_ind[3].num
    no_booms=fittest_ind[2].num
    no_large_skimmers=fittest_ind[1].num
    no_small_skimmers=fittest_ind[0].num
    print ("Maximum fitness achieved: "+ str(max(fitness)))
    print ("Number of small skimmers needed: "+ str(no_small_skimmers))
    print ("Number of large skimmers needed: "+ str(no_large_skimmers))
    print ("Number of booms needed: "+ str(no_booms))
    print ("Number of shore booms needed: "+ str(no_shore_booms))
    plt.figure() #Figure showing our actions for the 3 first days



# Keeping things organised
if __name__=='__main__':
    main()
