package generator;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;


/**
 * 
 * Klasa do odczytywania/zapisywania individuali z/do pliku
 * 
 * @author Kacper Gorski
 *
 */
public final class IndividualIO {

  
    public static void zapiszDoPliku(ArrayList<EvBinaryVectorIndividual> aOsobniki, String file) {
      
  
      ObjectOutputStream ois = null;
      try {
        ois = new ObjectOutputStream(new FileOutputStream(file));
      } catch (IOException e) {
        e.printStackTrace();
      }
      
      try {
        
        for (int i=0; i < aOsobniki.size(); i++) {
          aOsobniki.get(i).setObjectiveFunction(null);
          ois.writeObject(aOsobniki.get(i));
        }
        
      } catch (IOException e) {
        e.printStackTrace();
      }
      
    }
    
    
    public static  ArrayList<EvBinaryVectorIndividual> odczytajZPliku(String file) {
      
      ArrayList<EvBinaryVectorIndividual> pLista = new ArrayList<EvBinaryVectorIndividual>();
      File plik = new File(file);

      BufferedInputStream stream = null;
      
      try {
        stream = new BufferedInputStream(new FileInputStream(plik));
      } catch (FileNotFoundException e) {  e.printStackTrace(); }


      ObjectInputStream ois = null;
      try {
        ois = new ObjectInputStream(stream);
      } catch (IOException e) { e.printStackTrace(); }
      
      while (true) {
      
        EvBinaryVectorIndividual individual = null;
        try {
          individual = (EvBinaryVectorIndividual)ois.readObject();
          pLista.add(individual);
        } catch (Exception e) { break;  }
        
      }
      
      return pLista;
    }
    
    
    
    
  
}
