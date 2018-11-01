
plt.plot(lams, num_feats, 'o-')
plt.xscale('log')
plt.show()

plt.plot(lams, tprs, 'o-')
plt.plot(lams, fdrs, 'o-')
plt.xscale('log')
plt.show()



plt.plot(fdrs, tprs, 'o-')
plt.show()
