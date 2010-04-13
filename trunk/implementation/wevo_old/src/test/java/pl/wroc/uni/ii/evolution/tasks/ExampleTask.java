package pl.wroc.uni.ii.evolution.tasks;

import pl.wroc.uni.ii.evolution.distribution.strategies.EvIslandModel;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchangeWithServlet;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchanger;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiver;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiverImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvSender;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.ExSenderImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvRing;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopology;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvIslandDistribution;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvTaskCreator;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticDatabaseSuppportServletStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.EvPermutationTransposeMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvBestValueNotImproved;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationWithErrorRecovery;
import pl.wroc.uni.ii.evolution.solutionspaces.EvPermutationSpace;



   public class ExampleTask implements EvTaskCreator {
     
     public Runnable create(int task_id, long node_id, String wevo_url) {
           
       //objective function & algorithm section
       EvCycleNumber objective_function = new EvCycleNumber();
       EvAlgorithm<EvPermutationIndividual> genericEA = new EvAlgorithm<EvPermutationIndividual>(100);
       EvPermutationTransposeMutation mutation = new EvPermutationTransposeMutation(0.2);
       
       genericEA.setSolutionSpace(new EvPermutationSpace(30));
       genericEA.setObjectiveFunction(objective_function);
       genericEA.setTerminationCondition(new EvBestValueNotImproved<EvPermutationIndividual>(100));
       genericEA.addOperator(mutation);
       //genericEA.addOperator(cross);
       genericEA.addOperator(new EvTournamentSelection<EvPermutationIndividual>(8,7));
       
       //we use ring topology
       EvTopology topology = new EvRing(3); 
       
       
                       /******** HERE WE ADD STATISTICS OPERATOR ************/       
       long cell_id = topology.assignCellID();
       // CREATE STORAGE OBJECT (all data will be stored in central database)
       EvPersistentStatisticDatabaseSuppportServletStorage storage = 
         new EvPersistentStatisticDatabaseSuppportServletStorage(task_id, cell_id , node_id, wevo_url);
       // ADD STATISTICS OPERATOR TO ALGORITHM
       genericEA.addOperatorToEnd(new EvObjectiveFunctionValueMaxAvgMinGatherer<EvPermutationIndividual>(storage));        
              
       
       
       // THE REST OF EVISLAND STUFF
       /*
        * this is how it was before refactoring
       EvIndividualsExchanger<EvPermutationIndividual> exchanger = new EvIndividualsExchangeWithServlet<EvPermutationIndividual>(new EvDBServletCommunicationWithErrorRecovery(wevo_url, 5, 2000));
       EvReceiver<EvPermutationIndividual> receiver = new EvReceiverImpl<EvPermutationIndividual>(exchanger, topology,  20, 10000, task_id, 10, false); 
       EvSender<EvPermutationIndividual> sender = new ExSenderImpl<EvPermutationIndividual>(exchanger, topology, 20, 10000, task_id, node_id, 10,  false);       
       EvIslandModel<EvPermutationIndividual> strategy = new EvIslandModel<EvPermutationIndividual>(receiver, sender); 
       EvIslandDistribution task = new EvIslandDistribution(strategy, genericEA);
       */ 
       
       // and how it looks now:
       EvIslandModel<EvPermutationIndividual> model = 
         new EvIslandModel(wevo_url, topology, task_id, 10, 10000, 10, 10000, false, false);
       EvIslandDistribution task = new EvIslandDistribution(model, genericEA);
       
       return task;
     }
  }     
  
   
   
   class EvCycleNumber implements EvObjectiveFunction<EvPermutationIndividual> {


     /**
      * 
      */
     private static final long serialVersionUID = -7533937838475216360L;

     public double evaluate(EvPermutationIndividual individual) {
       int[] chromosome = individual.getChromosome().clone();
       int cycle_number = 0;
       int k;
       for(int i = 0; i < chromosome.length; i++) {
         k = i;
         
         if(chromosome[k] > -1){
           while(chromosome[k] != -1) {
             int j = k;
             k = chromosome[k];
             chromosome[j] = -1;
           }
           cycle_number++;
         }

       }
       return cycle_number;
     }

   }

