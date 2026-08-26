// id \t name \t engine speed literal \t tiles/s, for every registered troop.
#include "GameManager.h"
#include "CardRegistry.h"
#include "Troop.h"
#include "ClashEnv.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
int main(){
    auto& reg=CardRegistry::getInstance();
    std::vector<std::pair<int,CardDefinition>> all;
    for(auto&[id,def]:reg.getAllCards()) all.emplace_back(id,def);
    std::sort(all.begin(),all.end(),[](auto&a,auto&b){return a.first<b.first;});
    std::cout<<std::fixed<<std::setprecision(4);
    for(auto&[id,def]:all){
        if(def.isSpell) continue;
        Board b; def.spawnEntity(9,10,0,b); b.commitPendingEntities();
        for(auto&e:b.getEntities())
            if(auto* t=dynamic_cast<Troop*>(e.get())){
                std::cout<<id<<"\t"<<def.name<<"\t"<<t->getSpeed()<<"\t"
                         <<t->getSpeed()*10.0f<<"\t"<<(def.isEvolution?"E":"-")<<"\n";
                break;
            }
    }
}
