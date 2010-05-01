package generator;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class CustomGenerator {

	public static void main(String[] args) throws IOException {
		File bla = new File("/home/railman/workspace/svn/texas-cuda-ai1/implementation/cuda_texas/target/classes/texas_individuale/generacja3.bin");
		FileInputStream pBla = new FileInputStream(bla);
	
		DataInputStream adsa = new DataInputStream(pBla);
		int sra = adsa.readInt();
		System.out.println(sra);

	}
	
	private int seed=0;
	private int prev_seed=0;
	
	public CustomGenerator() {
		
	}
	
	public CustomGenerator(int seed) {
		
		this.seed = seed;
	}
	
	public int nextInt(int ograniczenie) {
		int wylosowana = seed;

		if (wylosowana < 0) 
			wylosowana=-wylosowana;


		seed = (seed * 12991 + 127)%12345789;
		
		return wylosowana%ograniczenie;
	}

	public int getSeed() {
		return seed;
	}
}
