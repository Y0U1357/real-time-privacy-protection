// Global minimum-cost assignment with optional unmatched rows/columns.
// Dummy costs mirror lapjv(extend_cost=True, cost_limit=threshold).
export function linearAssignment(costs, columns, threshold) {
  const rows = costs.length, n = rows + columns;
  if (!rows || !columns) return {matches:[], unmatchedRows:Array.from({length:rows},(_,i)=>i), unmatchedCols:Array.from({length:columns},(_,i)=>i)};
  const a = Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>
    i < rows && j < columns ? (costs[i][j] <= threshold ? costs[i][j] : 1e6) :
      i >= rows && j >= columns ? 0 : threshold/2));
  const u=Array(n+1).fill(0), v=Array(n+1).fill(0), p=Array(n+1).fill(0), way=Array(n+1).fill(0);
  for (let i=1;i<=n;i++) {
    p[0]=i; let j0=0;
    const minv=Array(n+1).fill(Infinity), used=Array(n+1).fill(false);
    do {
      used[j0]=true; const i0=p[j0]; let delta=Infinity,j1=0;
      for (let j=1;j<=n;j++) if (!used[j]) {
        const cur=a[i0-1][j-1]-u[i0]-v[j];
        if (cur<minv[j]) {minv[j]=cur;way[j]=j0;}
        if (minv[j]<delta) {delta=minv[j];j1=j;}
      }
      for (let j=0;j<=n;j++) if (used[j]) {u[p[j]]+=delta;v[j]-=delta;} else minv[j]-=delta;
      j0=j1;
    } while (p[j0]!==0);
    do {const j1=way[j0];p[j0]=p[j1];j0=j1;} while(j0!==0);
  }
  const matches=[];
  for(let j=1;j<=columns;j++) if(p[j]>0 && p[j]<=rows && costs[p[j]-1][j-1]<=threshold) matches.push([p[j]-1,j-1]);
  const mr=new Set(matches.map(v=>v[0])),mc=new Set(matches.map(v=>v[1]));
  return {matches,unmatchedRows:Array.from({length:rows},(_,i)=>i).filter(i=>!mr.has(i)),unmatchedCols:Array.from({length:columns},(_,i)=>i).filter(i=>!mc.has(i))};
}
