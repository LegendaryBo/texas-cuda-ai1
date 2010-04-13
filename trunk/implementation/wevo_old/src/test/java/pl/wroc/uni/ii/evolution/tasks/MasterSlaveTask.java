package pl.wroc.uni.ii.evolution.tasks;


import pl.wroc.uni.ii.evolution.distribution.eval.EvExternalEvaluationOperator;
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
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvLongFunction;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationWithErrorRecovery;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

public class MasterSlaveTask implements EvTaskCreator {
    public Runnable create(int task_id, long node_id,
        String wevo_server_url) {

    int bits = 50;
    int pop_size=100;
    EvTopology topology = new EvRing(3); 
    
    EvAlgorithm<EvBinaryVectorIndividual> alg = new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);
    EvLongFunction<EvBinaryVectorIndividual> one_max = new EvLongFunction<EvBinaryVectorIndividual>(new EvOneMax(), 100);
    EvBinaryVectorSpace solution_space = new EvBinaryVectorSpace(one_max, bits);
    alg.setSolutionSpace(solution_space);
    alg.addOperatorToEnd(new EvBinaryVectorNegationMutation(0.02));
    alg.addOperator(new EvExternalEvaluationOperator<EvBinaryVectorIndividual>(200, task_id, node_id, wevo_server_url));
    alg.addOperatorToEnd(new EvTournamentSelection<EvBinaryVectorIndividual>(4,2));
    alg.addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(System.out));
    alg.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(8));    


    // THE REST OF EVISLAND STUFF
    EvIndividualsExchanger<EvBinaryVectorIndividual> exchanger = new EvIndividualsExchangeWithServlet<EvBinaryVectorIndividual>(new EvDBServletCommunicationWithErrorRecovery(wevo_server_url, 5, 2000));
    EvReceiver<EvBinaryVectorIndividual> receiver = new EvReceiverImpl<EvBinaryVectorIndividual>(exchanger, topology,  20, 10000, task_id, 10, false); 
    EvSender<EvBinaryVectorIndividual> sender = new ExSenderImpl<EvBinaryVectorIndividual>(exchanger, topology, 20, 10000, task_id, node_id, 10,  false);
    EvIslandModel<EvBinaryVectorIndividual> strategy = new EvIslandModel<EvBinaryVectorIndividual>(receiver, sender); 
    EvIslandDistribution task = new EvIslandDistribution(strategy, alg);
    
    return task;    
    
    
    }  
    
  }


