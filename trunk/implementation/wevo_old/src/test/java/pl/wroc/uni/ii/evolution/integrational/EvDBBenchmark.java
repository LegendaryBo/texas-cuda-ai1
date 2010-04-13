package pl.wroc.uni.ii.evolution.integrational;

import java.io.IOException;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import junit.framework.TestCase;

public class EvDBBenchmark extends TestCase {
  
	public void testDummy() {
		
	}
	
//  private String WEVO_SERVER_ADDRESS = "http://127.0.0.1:8080";
//
//  private EvDBServletCommunication db;
//  
//  private int task_id = 12345;
//  private int cell_id = 13;
//  private int node_id = 6666;
//
//  @Override
//  protected void setUp() throws Exception {
//    db = new EvDBServletCommunicationImpl(WEVO_SERVER_ADDRESS);
//  }
//  
//  
//  public void testAddIndividuals() throws IOException {
//    
//    int n = 3;
//    EvBinaryVectorIndividual ind = new EvBinaryVectorIndividual(n);
//   
//    long start = System.currentTimeMillis();
//    
//    System.out.println("Delete: start");
//    db.deleteIndividualsFromTask(task_id);
//    System.out.println("Delete: end");
//    
//    for (int i = 0 ; i < n; i++) {
//      db.addIndividual(ind, task_id, 0, cell_id, node_id);
//    }
//    long end = System.currentTimeMillis();
//   
//    System.out.println("Wys�anie " + n + " osobnik�w: " +  (end - start) + " ms");
//   
//  }
//  
//  public void testAddIndividualsBetter() throws IOException {
//    
//    
//    db.deleteStatisticForTask(12345);
//    
//   
//    int n = 2000;
//    EvBinaryVectorIndividual[] ind = new EvBinaryVectorIndividual[n];
//    double[] values = new double[n]; 
//    
//    for (int i = 0; i < n; i++) {
//      ind[i] = new EvBinaryVectorIndividual(100);
//      values[i] = i;
//    }
//    
//    
//    db.deleteIndividualsFromTask(task_id);
//    long start = System.currentTimeMillis();
//    db.addIndividuals(ind, task_id, values, cell_id, node_id); 
//    long end = System.currentTimeMillis();
//    
//    System.out.println("Better: Wys�anie " + n + " osobnik�w: " +  (end - start) + " ms");
//   
//  }

}
