'''=================================================================================================
Required Libraries
================================================================================================='''
import numpy as np
import matplotlib.pyplot as plt
'''=================================================================================================
Base classes
================================================================================================='''
# Prestige oil outflow happened 130 miles from the shore of Galicia, so only booms and skimmers are gonna be used
class skimmers:
    def __init__(self,vol_inflow,capacity):
        self.vi=vol_inflow #m^3/h
        self.cap=capacity #m^3
        self.num=np.random.randint(0,300) # how many we have
    def use(self):
        if (self.num!=0):
            self.num-=1

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
xremain=np.array([])
yremain=np.array([])
xdisp=np.array([])
ydisp=np.array([])
remaining_curve=np.polyfit(xremain,yremain,2) #2nd degree polyonymials are chosen for the curves
disperse_curve=np.polyfit(xdisp,ydisp,2)

#Goal for the first 72 hours
xremain_std=np.array([])
yremain_std=np.array([])
xdisp_std=np.array([])
ydisp_std=np.array([])
remaining_curve_std=np.polyfit(xremain_std,yremain_std,2) #2nd degree polyonymials are chosen for the curves
disperse_curve_std=np.polyfit(xdisp_std,ydisp_std,2)
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
    time_grid=np.linspace(0,72,72) #Render grid to compute losses and limitation curves
    rem_control_metric=None
    disp_control_metric=None
    return rem_control_metric, disp_control_metric

def render_map():
    pass

'''================================================================================================
Main function
================================================================================================'''
def main():
    # Genetic Algorithm
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
    render_map() # Rendering of equipment placing on a real map


# Keeping things organised
if __name__=='__main__':
    main()
